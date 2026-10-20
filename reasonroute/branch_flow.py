"""Existing per-branch flow, retained for the method runtime."""
import os
from typing import Dict, Tuple
from reasonroute.retrieval import InterleavedRetrieval as IRCoTBaseline
from tools.rag_resarch_tool import DeepSeekRAGTool, HybridQuestionClassifier

class ReasonRouteIRCoTAssisted:
    """Online route: lightweight ReasonRoute for fact, IRCoT for complex questions."""

    def __init__(
        self,
        rr_engine: DeepSeekRAGTool,
        ircot_engine: IRCoTBaseline,
        mode: str = "exp_reason",
        route_source: str = "predicted",
        classifier: HybridQuestionClassifier = None,
    ):
        if route_source != 'predicted':
            raise ValueError('This method helper only supports predicted routing; legacy gold experiments are separate')
        self.rr_engine = rr_engine
        self.ircot_engine = ircot_engine
        self.mode = mode
        self.route_source = route_source
        self.classifier = classifier

    @staticmethod
    def _answer_is_bad(answer: str) -> bool:
        text = (answer or "").strip()
        low = text.lower()
        if not text:
            return True
        bad_markers = (
            "search(",
            "unknown",
            "insufficient",
            "not enough",
            "资料不足",
            "证据不足",
            "无法确定",
        )
        if any(marker in low or marker in text for marker in bad_markers):
            return True
        return len(text) > int(os.getenv("CHEAP_COMPLEX_LONG_ANSWER_CHARS", "180"))

    @staticmethod
    def _answer_has_failure_marker(answer: str) -> bool:
        text = (answer or "").strip()
        low = text.lower()
        if not text:
            return True
        bad_markers = (
            "search(",
            "unknown",
            "insufficient",
            "not enough",
            "资料不足",
            "证据不足",
            "无法确定",
        )
        return any(marker in low or marker in text for marker in bad_markers)

    def route_type(self, query: str, gold_question_type: str) -> Tuple[str, str]:
        if gold_question_type:
            raise ValueError('Do not pass gold labels into the retrieval flow')
        if self.classifier is None:
            raise RuntimeError("Predicted routing requires a classifier.")
        predicted = self.classifier.classify(query)
        if predicted not in {"fact", "explanation", "reasoning"}:
            predicted = "explanation"
        return predicted, predicted

    def select_branch(self, question_type: str) -> str:
        if self.mode == "always_fact":
            return "reasonroute_fact"
        if self.mode == "always_explanation":
            return "ircot_explanation"
        if self.mode == "always_reasoning":
            if os.getenv("ALWAYS_REASONING_DIRECT_IRCOT", "0").strip().lower() not in {"0", "false", "no"}:
                return "ircot"
            return "reasonroute_reasoning"
        if self.mode == "exp":
            return "ircot" if question_type == "explanation" else "reasonroute_fact"
        if self.mode == "exp_reason":
            return "ircot" if question_type in {"explanation", "reasoning"} else "reasonroute_fact"
        if self.mode == "exp_ircot_reason_rr":
            if question_type == "explanation":
                return "ircot"
            if question_type == "reasoning":
                return "reasonroute_reasoning"
            return "reasonroute_fact"
        raise ValueError(f"Unknown assisted mode: {self.mode}")

    def run_cheap_to_complex_reasoning(self, query: str) -> Tuple[Dict, str]:
        cheap_context_top_k = os.getenv("REASONING_CHEAP_FACT_CONTEXT_TOP_K")
        cheap_gen_max_tokens = os.getenv("REASONING_CHEAP_FACT_GEN_MAX_TOKENS")
        first = self.rr_engine.execute_rag_flow(
            query,
            "fact",
            answer_style="reasoning",
            context_top_k_override=int(cheap_context_top_k) if cheap_context_top_k else None,
            gen_max_tokens_override=int(cheap_gen_max_tokens) if cheap_gen_max_tokens else None,
        )
        first_metrics = first.get("metrics", {})
        if not self._answer_is_bad(first.get("final_answer", "")):
            first_metrics["cheap_to_complex_route"] = "fact_only"
            return first, "cheap_to_complex_fact"

        fallback_mode = os.getenv("CHEAP_COMPLEX_FALLBACK", "ircot").strip().lower()
        if fallback_mode in {"rr", "rr_reasoning", "reasonroute_reasoning"}:
            second = self.rr_engine.execute_rag_flow(query, "reasoning", answer_style="reasoning")
            fallback_branch = "cheap_to_complex_rr_reasoning"
        else:
            second = self.ircot_engine.run(query, enable_answer_verifier=True)
            fallback_branch = "cheap_to_complex_ircot"
        second_metrics = second.get("metrics", {})
        merged_metrics = dict(second_metrics)
        merged_metrics["retrieval_steps"] = first_metrics.get("retrieval_steps", 0) + second_metrics.get("retrieval_steps", 0)
        merged_metrics["context_length"] = first_metrics.get("context_length", 0) + second_metrics.get("context_length", 0)
        merged_metrics["token_cost"] = first_metrics.get("token_cost", 0) + second_metrics.get("token_cost", 0)
        merged_metrics["cheap_to_complex_route"] = fallback_branch
        second["metrics"] = merged_metrics
        return second, fallback_branch

    def run_forced_cheap_reasoning(self, query: str) -> Tuple[Dict, str]:
        """Run only the compact first stage of the cheap-to-complex branch."""
        context_top_k = int(os.getenv("REASONING_CHEAP_FACT_CONTEXT_TOP_K", "4"))
        gen_max_tokens = int(os.getenv("REASONING_CHEAP_FACT_GEN_MAX_TOKENS", "80"))
        output = self.rr_engine.execute_rag_flow(
            query,
            "fact",
            answer_style="reasoning",
            context_top_k_override=context_top_k,
            gen_max_tokens_override=gen_max_tokens,
        )
        metrics = dict(output.get("metrics", {}))
        metrics["cheap_to_complex_route"] = "forced_fact_only"
        output["metrics"] = metrics
        return output, "compact_reasoning_fact_only"

    def maybe_repair_bad_answer(self, query: str, output: Dict) -> Dict:
        enabled = os.getenv("SELECTIVE_BAD_ANSWER_REPAIR", "0").strip().lower() not in {"0", "false", "no"}
        if not enabled or not self._answer_is_bad(output.get("final_answer", "")):
            return output

        context = output.get("retrieved_context", [])
        if not context:
            return output
        top_k = int(os.getenv("SELECTIVE_REPAIR_CONTEXT_TOP_K", "3"))
        repaired, repair_tokens = self.ircot_engine.verify_final_answer(
            query,
            context[:top_k],
            output.get("final_answer", ""),
        )
        metrics = dict(output.get("metrics", {}))
        metrics["token_cost"] = metrics.get("token_cost", 0) + repair_tokens
        metrics["selective_answer_repair"] = True
        metrics["selective_answer_repair_tokens"] = repair_tokens
        output["metrics"] = metrics
        if repaired:
            output["final_answer"] = repaired
        return output

    def finalize_output(
        self,
        query: str,
        output: Dict,
        branch: str,
        route_type: str,
        predicted_type: str,
    ) -> Tuple[Dict, str, str, str]:
        return self.maybe_repair_bad_answer(query, output), branch, route_type, predicted_type

    def run(self, query: str, question_type: str) -> Tuple[Dict, str, str, str]:
        route_type, predicted_type = self.route_type(query, question_type)
        branch = self.select_branch(route_type)
        if branch in {"ircot", "ircot_explanation"}:
            if (
                branch == "ircot"
                and route_type == "reasoning"
                and os.getenv("REASONING_CHEAP_TO_COMPLEX", "0").strip().lower() not in {"0", "false", "no"}
            ):
                output, actual_branch = self.run_cheap_to_complex_reasoning(query)
                return self.finalize_output(query, output, actual_branch, route_type, predicted_type)
            enable_verifier = (
                branch == "ircot"
                and (route_type == "reasoning" or self.mode == "always_reasoning")
                and os.getenv("REASONING_IRCOT_ANSWER_VERIFIER", "0").strip().lower() not in {"0", "false", "no"}
            )
            if branch == "ircot_explanation" or (branch == "ircot" and route_type == "explanation"):
                explanation_branch = os.getenv("EXPLANATION_BRANCH_MODE", "ircot").strip().lower()
                if explanation_branch in {"reasonroute", "reasonroute_context", "context_enhanced"}:
                    context_top_k = int(os.getenv("EXPLANATION_RR_CONTEXT_TOP_K", "4"))
                    gen_max_tokens = int(os.getenv("EXPLANATION_RR_GEN_MAX_TOKENS", "100"))
                    output = self.rr_engine.execute_rag_flow(
                        query,
                        "explanation",
                        answer_style="fact",
                        context_top_k_override=context_top_k,
                        gen_max_tokens_override=gen_max_tokens,
                    )
                    return self.finalize_output(
                        query,
                        output,
                        "reasonroute_context_explanation",
                        route_type,
                        predicted_type,
                    )
                if os.getenv("EXPLANATION_CONDITIONAL_FALLBACK", "0").strip().lower() not in {"0", "false", "no"}:
                    primary = self.ircot_engine.run(query, enable_answer_verifier=False)
                    fallback_policy = os.getenv("EXPLANATION_FALLBACK_POLICY", "failure_marker").strip().lower()
                    primary_failed = (
                        self._answer_is_bad(primary.get("final_answer", ""))
                        if fallback_policy in {"bad", "bad_or_long", "legacy"}
                        else self._answer_has_failure_marker(primary.get("final_answer", ""))
                    )
                    if not primary_failed:
                        return self.finalize_output(
                            query,
                            primary,
                            "ircot_explanation_primary",
                            route_type,
                            predicted_type,
                        )

                    context_top_k = int(os.getenv("EXPLANATION_RR_CONTEXT_TOP_K", "4"))
                    gen_max_tokens = int(os.getenv("EXPLANATION_RR_GEN_MAX_TOKENS", "100"))
                    fallback = self.rr_engine.execute_rag_flow(
                        query,
                        "explanation",
                        answer_style="fact",
                        context_top_k_override=context_top_k,
                        gen_max_tokens_override=gen_max_tokens,
                        expansion_seed_context=primary.get('retrieved_context', []),
                    )
                    use_fallback = True
                    selected, extra = fallback, primary
                    selected_metrics = dict(selected.get("metrics", {}))
                    extra_metrics = extra.get("metrics", {})
                    for metric_key in ("retrieval_steps", "token_cost"):
                        selected_metrics[metric_key] = (
                            selected_metrics.get(metric_key, 0) + extra_metrics.get(metric_key, 0)
                        )
                    selected_metrics["explanation_conditional_fallback"] = True
                    selected_metrics["explanation_fallback_selected"] = use_fallback
                    selected_metrics["explanation_primary_token_cost"] = primary.get("metrics", {}).get("token_cost", 0)
                    selected_metrics["explanation_fallback_token_cost"] = fallback.get("metrics", {}).get("token_cost", 0)
                    selected_metrics["explanation_fallback_policy"] = fallback_policy
                    selected["metrics"] = selected_metrics
                    return self.finalize_output(
                        query,
                        selected,
                        "ircot_then_context_explanation" if use_fallback else "ircot_explanation_retained",
                        route_type,
                        predicted_type,
                    )
                enable_verifier = (
                    os.getenv("EXPLANATION_IRCOT_ANSWER_VERIFIER", "0").strip().lower()
                    not in {"0", "false", "no"}
                )
            output = self.ircot_engine.run(query, enable_answer_verifier=enable_verifier)
            return self.finalize_output(query, output, branch, route_type, predicted_type)
        if branch == "reasonroute_reasoning":
            if os.getenv("REASONING_FORCE_CHEAP_FACT", "0").strip().lower() not in {"0", "false", "no"}:
                output, actual_branch = self.run_forced_cheap_reasoning(query)
                return self.finalize_output(query, output, actual_branch, route_type, predicted_type)
            if os.getenv("REASONING_CHEAP_TO_COMPLEX", "0").strip().lower() not in {"0", "false", "no"}:
                output, actual_branch = self.run_cheap_to_complex_reasoning(query)
                return self.finalize_output(query, output, actual_branch, route_type, predicted_type)
            output = self.rr_engine.execute_rag_flow(query, "reasoning", answer_style="reasoning")
            return self.finalize_output(query, output, branch, route_type, predicted_type)
        if os.getenv("FACT_ADAPTIVE_QUESTION_LENGTH", "0").strip().lower() not in {"0", "false", "no"}:
            threshold = int(os.getenv("FACT_ADAPTIVE_LENGTH_THRESHOLD", "50"))
            use_extended_budget = len(query.strip()) >= threshold
            context_top_k = int(
                os.getenv("FACT_ADAPTIVE_LONG_TOP_K", "4")
                if use_extended_budget
                else os.getenv("FACT_ADAPTIVE_SHORT_TOP_K", "3")
            )
            gen_max_tokens = int(
                os.getenv("FACT_ADAPTIVE_LONG_GEN_MAX_TOKENS", "80")
                if use_extended_budget
                else os.getenv("FACT_ADAPTIVE_SHORT_GEN_MAX_TOKENS", "60")
            )
            output = self.rr_engine.execute_rag_flow(
                query,
                "fact",
                answer_style="fact",
                context_top_k_override=context_top_k,
                gen_max_tokens_override=gen_max_tokens,
            )
            budget_label = "long" if use_extended_budget else "short"
            return self.finalize_output(
                query,
                output,
                f"reasonroute_fact_adaptive_{budget_label}",
                route_type,
                predicted_type,
            )
        output = self.rr_engine.execute_rag_flow(query, "fact", answer_style="fact")
        return self.finalize_output(query, output, branch, route_type, predicted_type)
