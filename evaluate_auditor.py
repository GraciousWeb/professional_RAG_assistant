import asyncio
import os
import glob
import csv
from dotenv import load_dotenv
from pydantic import BaseModel

from openai import AsyncOpenAI

from ragas import Dataset, experiment
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from retrieval import get_compliance_chain

load_dotenv()


DATA_ROOT = "./ragas_data"
DATASET_NAME = "iso_audit_dataset"
EXPERIMENT_NAME = "iso_audit_initial_run"

METRIC_COLS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

os.makedirs(DATA_ROOT, exist_ok=True)

async_client = AsyncOpenAI()

eval_llm = llm_factory("gpt-4o-mini", client=async_client)

eval_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    client=async_client, 
)

faithfulness = Faithfulness(llm=eval_llm)
answer_rel = AnswerRelevancy(llm=eval_llm, embeddings=eval_embeddings)
ctx_precision = ContextPrecision(llm=eval_llm)
ctx_recall = ContextRecall(llm=eval_llm)


class AuditResult(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


@experiment(experiment_model=AuditResult)
async def compliance_audit_eval(row: dict):
    user_input = row["user_input"]
    response = row["response"]
    retrieved_contexts = row["retrieved_contexts"]
    reference = row["reference"]

    f = await faithfulness.ascore(
        user_input=user_input,
        response=response,
        retrieved_contexts=retrieved_contexts,
    )
    ar = await answer_rel.ascore(
        user_input=user_input,
        response=response,
    )
    cp = await ctx_precision.ascore(
        user_input=user_input,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )
    cr = await ctx_recall.ascore(
        user_input=user_input,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )

    return AuditResult(
        faithfulness=float(f.value),
        answer_relevancy=float(ar.value),
        context_precision=float(cp.value),
        context_recall=float(cr.value),
    )


def _find_latest_experiment_csv(root_dir: str, experiment_name: str) -> str:
    experiments_dir = os.path.join(root_dir, "experiments")
    pattern = os.path.join(experiments_dir, f"*{experiment_name}*.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No experiment CSV found matching: {pattern}\n"
            f"Check {experiments_dir} for outputs."
        )
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def print_aggregates_from_csv(csv_path: str):
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []

        metric_values = {k: [] for k in METRIC_COLS if k in cols}

        for row in reader:
            for k in metric_values.keys():
                metric_values[k].append(_safe_float(row.get(k)))

    print("\n" + "=" * 30)
    print("FINAL AUDITOR PERFORMANCE (mean over samples)")
    print("=" * 30)

    if not metric_values:
        print("No metric columns found in CSV.")
        print("CSV columns were:", cols)
        return

    for k, vals in metric_values.items():
        m = _mean(vals)
        print(f"{k}: {m:.4f}" if m is not None else f"{k}: NaN")


async def main():
    chain = get_compliance_chain()
    print("Running chain & preparing dataset rows...")

    questions = [
        "What are the requirements for the information security policy according to ISO/IEC 27001:2022 clause 5.2",
        "What does ISO/IEC 27001:2022 require an organization to do regarding internal audits?",
    ]
    ground_truths = [
        "Top management shall establish an information security policy that is appropriate to the organization, includes information security objectives, commits to satisfying applicable information security requirements, and commits to continual improvement of the information security management system.",
        "The organization shall conduct internal audits at planned intervals to determine whether the information security management system conforms to organizational and ISO/IEC 27001 requirements and is effectively implemented and maintained.",
    ]

    dataset = Dataset(
        name=DATASET_NAME,
        backend="local/csv",
        root_dir=DATA_ROOT,
    )

    for i, (q, gt) in enumerate(zip(questions, ground_truths), start=1):
        result = chain.invoke(q)
        dataset.append(
            {
                "id": f"sample_{i}",
                "user_input": q,
                "response": result["answer"],
                "retrieved_contexts": [doc.page_content for doc in result["sources"]],
                "reference": gt,
            }
        )

    dataset.save()

    print("Starting Ragas Experiment...")
    _ = await compliance_audit_eval.arun(dataset, name=EXPERIMENT_NAME)

    csv_path = _find_latest_experiment_csv(DATA_ROOT, EXPERIMENT_NAME)
    print(f"Experiment results saved to: {csv_path}")
    print_aggregates_from_csv(csv_path)

if __name__ == "__main__":
    asyncio.run(main())