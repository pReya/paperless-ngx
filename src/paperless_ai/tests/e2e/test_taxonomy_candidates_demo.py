"""End-to-end demonstration: existing taxonomy that the similar documents
don't carry is invisible to the AI suggestions.

Everything below runs for real against the test database: a library of
documents, a classifier trained with scikit-learn on that library, the
built-in matching rules, neighbour-derived candidate building, prompt
rendering and the response validation. Only two things are replaced:

- ``retrieve_similar_nodes`` (the vector store) returns a chosen set of real
  documents as the nearest neighbours, so the scenario is deterministic.
- ``AIClient.run_llm_query`` is a scripted stand-in for the model. It reads
  the candidate block from the real prompt and behaves like an obedient
  model following the prompt's own instructions: a name that is offered as a
  candidate is reused and paired with its id; a concept with no matching
  candidate is written out as a fresh name. That is the behaviour #13676 /
  #13866 were built around, and the reason near-duplicate names appear.

Run on ``dev`` these tests fail and their assertion messages show what the
model was shown and what came back. Run on the feature branch they pass.
See ``e2e/README.md`` for the before/after runner.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any

import pytest

from documents.classifier import DocumentClassifier
from documents.models import Document
from documents.models import MatchingModel
from paperless_ai.ai_classifier import get_ai_document_classification
from paperless_ai.base_model import DocumentClassifierSchema
from paperless_ai.base_model import model_to_classification_suggestions
from paperless_testing.factories import CorrespondentFactory
from paperless_testing.factories import DocumentFactory
from paperless_testing.factories import TagFactory
from paperless_testing.factories import UserFactory

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest_mock

# --------------------------------------------------------------------------- #
# Scenario: a medical archive
# --------------------------------------------------------------------------- #

BLOODWORK_TEXTS = [
    f"Laboratory report. Complete blood count: hemoglobin {13 + i % 3} g/dL, leukocytes "
    f"{5 + i % 4} /nL, thrombocytes {200 + 10 * i} /nL. Cholesterol {170 + 5 * i} mg/dL. Sample analysed by "
    "MedLab Diagnostics. All values within reference range."
    for i in range(14)
]

CAR_TEXTS = [
    "Workshop invoice. Vehicle inspection, oil change, brake pads replaced, "
    f"tyre rotation. Mileage {80000 + 1500 * i} km. Next service due in 12 "
    "months. Thank you for choosing our garage."
    for i in range(14)
]

# Older scans that were bulk-imported and never tagged. The stand-in vector
# store returns these as the new document's nearest neighbours.
UNTAGGED_LAB_TEXTS = [
    "Lab results summary for patient visit. Blood count and lipid panel "
    f"ordered by general practitioner, visit {i + 1}. Hemoglobin normal, "
    "lipids borderline. Please discuss at next appointment."
    for i in range(6)
]

NEW_DOCUMENT_TEXT = (
    "Laboratory report. Complete blood count: hemoglobin 14 g/dL, leukocytes "
    "6 /nL, thrombocytes 250 /nL. Cholesterol 210 mg/dL. Sample analysed by "
    "MedLab Diagnostics. Cholesterol slightly above reference range."
)


@pytest.fixture
def archive(paperless_dirs):
    """A library with a trained classifier and one matching rule.

    - ``Bloodwork`` (Auto) and ``Car`` (Auto) tags, 14 documents each - the
      classifier learns both.
    - ``MedLab Diagnostics`` correspondent with a literal matching rule.
    - Six untagged lab summaries that will be the new document's neighbours.
    - The new document itself, carrying only the inbox tag.
    """
    inbox = TagFactory.create(name="Inbox", is_inbox_tag=True)
    bloodwork = TagFactory.create(
        name="Bloodwork",
        matching_algorithm=MatchingModel.MATCH_AUTO,
    )
    car = TagFactory.create(name="Car", matching_algorithm=MatchingModel.MATCH_AUTO)
    medlab = CorrespondentFactory.create(
        name="MedLab Diagnostics",
        match="medlab",
        matching_algorithm=MatchingModel.MATCH_LITERAL,
    )

    for text in BLOODWORK_TEXTS:
        DocumentFactory.create(content=text, correspondent=medlab).tags.add(bloodwork)
    for text in CAR_TEXTS:
        DocumentFactory.create(content=text).tags.add(car)
    neighbours = [DocumentFactory.create(content=text) for text in UNTAGGED_LAB_TEXTS]

    new_document = DocumentFactory.create(
        title="Lab results 2026-03",
        content=NEW_DOCUMENT_TEXT,
    )
    new_document.tags.add(inbox)

    classifier = DocumentClassifier()
    assert classifier.train() is True
    classifier.save()

    return SimpleNamespace(
        bloodwork=bloodwork,
        car=car,
        medlab=medlab,
        neighbours=neighbours,
        new_document=new_document,
    )


# --------------------------------------------------------------------------- #
# Scripted model
# --------------------------------------------------------------------------- #

# What an attentive reader would take away from NEW_DOCUMENT_TEXT, as
# (category, concept the model has in mind, name it writes when nothing is
# offered, offered candidate name it would recognise as the same thing).
CONCEPTS = [
    ("tags", "blood test", "Bloodwork"),
    ("correspondents", "MedLab Diagnostics GmbH", "MedLab Diagnostics"),
]


def _candidates_in_prompt(prompt: str) -> dict[str, dict[str, int]]:
    """Parse the ``{"tags": [{"id":..,"name":..}], ...}`` block the real
    prompt renders, as name -> id per category. Empty when no block."""
    match = re.search(r"\(untrusted data\):\n(\{.*?\})\n", prompt, re.DOTALL)
    if match is None:
        return {}
    payload = json.loads(match.group(1))
    return {
        category: {c["name"]: c["id"] for c in entries}
        for category, entries in payload.items()
    }


def scripted_model(seen: list[dict[str, Any]]) -> Callable[..., Any]:
    """Build a ``run_llm_query`` replacement that records every call."""

    def run_llm_query(prompt: str, *, allowed_candidate_ids=None):
        offered = _candidates_in_prompt(prompt)
        answer: dict[str, Any] = {
            "title": "Lab results summary",
            "dates": ["2026-03-01"],
        }
        for category, invented, recognised in CONCEPTS:
            singular = category[:-1]
            if recognised in offered.get(category, {}):
                answer[category] = [recognised]
                answer[f"matched_{category}"] = [recognised]
                answer[f"{singular}_ids"] = [offered[category][recognised]]
            else:
                answer[category] = [invented]
        schema = DocumentClassifierSchema(**answer)
        seen.append({"offered": offered, "answer": answer})
        # Same validation the real client applies to a model response.
        return model_to_classification_suggestions(schema, allowed_candidate_ids)

    return run_llm_query


@pytest.fixture
def model_calls(mocker: pytest_mock.MockerFixture) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    mocker.patch(
        "paperless_ai.client.AIClient.run_llm_query",
        side_effect=scripted_model(calls),
    )
    return calls


def neighbours_are(mocker: pytest_mock.MockerFixture, documents: list[Document]):
    nodes = [
        SimpleNamespace(metadata={"document_id": str(doc.pk)}, score=0.9 - 0.03 * i)
        for i, doc in enumerate(documents)
    ]
    return mocker.patch(
        "paperless_ai.ai_classifier.retrieve_similar_nodes",
        return_value=nodes,
    )


def explain(calls: list[dict[str, Any]], result) -> str:
    return (
        f"\n  candidates shown to the model: {json.dumps(calls[0]['offered'])}"
        f"\n  model answered:                {json.dumps(calls[0]['answer'])}"
        f"\n  suggestions after validation:  tags={result['tags']} "
        f"correspondents={result['correspondents']}"
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def ai_settings(settings) -> None:
    settings.LLM_BACKEND = "ollama"
    settings.LLM_MODEL = "some_model"
    settings.LLM_EMBEDDING_BACKEND = "huggingface"


@pytest.mark.django_db
class TestExistingTaxonomyIsOffered:
    def test_classifier_known_tag_missing_from_neighbours(
        self,
        archive,
        model_calls,
        mocker,
    ) -> None:
        """
        GIVEN:
            - 14 documents tagged "Bloodwork" exist and the classifier was
              trained on them
            - The new document's nearest neighbours are untagged imports
        WHEN:
            - AI suggestions are generated
        THEN:
            - "Bloodwork" is offered to the model and comes back as the
              existing tag, not as an invented "blood test"
        """
        neighbours_are(mocker, archive.neighbours)

        result = get_ai_document_classification(archive.new_document, user=None)

        why = explain(model_calls, result)
        assert "Bloodwork" in model_calls[0]["offered"].get("tags", {}), why
        assert result["tags"]["existing_ids"] == [archive.bloodwork.pk], why
        assert "blood test" not in result["tags"]["new_names"], why

    def test_rule_matched_correspondent_missing_from_neighbours(
        self,
        archive,
        model_calls,
        mocker,
    ) -> None:
        """
        GIVEN:
            - Correspondent "MedLab Diagnostics" has a literal matching rule
              that fires on the new document
            - None of the neighbours have a correspondent
        WHEN:
            - AI suggestions are generated
        THEN:
            - The correspondent is offered and reused instead of a new
              "MedLab Diagnostics GmbH" being suggested
        """
        neighbours_are(mocker, archive.neighbours)

        result = get_ai_document_classification(archive.new_document, user=None)

        why = explain(model_calls, result)
        assert result["correspondents"]["existing_ids"] == [archive.medlab.pk], why
        assert result["correspondents"]["new_names"] == [], why

    def test_candidates_survive_retrieval_outage(
        self,
        archive,
        model_calls,
        mocker,
    ) -> None:
        """
        GIVEN:
            - The vector store is unavailable
        WHEN:
            - AI suggestions are generated
        THEN:
            - Classifier and matching-rule hits are still offered, so the
              model reuses the existing tag and correspondent
        """
        mocker.patch(
            "paperless_ai.ai_classifier.retrieve_similar_nodes",
            side_effect=RuntimeError("vector store unavailable"),
        )

        result = get_ai_document_classification(archive.new_document, user=None)

        why = explain(model_calls, result)
        assert result["tags"]["existing_ids"] == [archive.bloodwork.pk], why
        assert result["correspondents"]["existing_ids"] == [archive.medlab.pk], why


@pytest.mark.django_db
class TestUnchangedBehaviour:
    """Controls that hold before and after the change."""

    def test_neighbour_carried_tag_is_still_offered(
        self,
        archive,
        model_calls,
        mocker,
    ) -> None:
        """
        GIVEN:
            - The nearest neighbours carry an "Imported" tag the classifier
              does not know (not Auto)
        WHEN:
            - AI suggestions are generated
        THEN:
            - "Imported" is still offered to the model, exactly as before
        """
        imported = TagFactory.create(name="Imported")
        for doc in archive.neighbours:
            doc.tags.add(imported)
        neighbours_are(mocker, archive.neighbours)

        get_ai_document_classification(archive.new_document, user=None)

        assert "Imported" in model_calls[0]["offered"]["tags"]

    def test_restricted_user_never_sees_hidden_taxonomy(
        self,
        archive,
        model_calls,
        mocker,
    ) -> None:
        """
        GIVEN:
            - The "Bloodwork" tag and the correspondent are owned by someone
              else, so a plain user may not view them
        WHEN:
            - AI suggestions are generated for that user
        THEN:
            - Neither is offered, whatever the classifier and rules say
        """
        owner = UserFactory.create()
        for obj in (archive.bloodwork, archive.medlab):
            obj.owner = owner
            obj.save()
        user = UserFactory.create()
        neighbours_are(mocker, archive.neighbours)

        result = get_ai_document_classification(archive.new_document, user=user)

        offered = model_calls[0]["offered"]
        assert "Bloodwork" not in offered.get("tags", {})
        assert "MedLab Diagnostics" not in offered.get("correspondents", {})
        assert result["tags"]["existing_ids"] == []
        assert result["correspondents"]["existing_ids"] == []
