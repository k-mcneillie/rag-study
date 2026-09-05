"""Assembling retrieved chunks into a prompt context.

This is where the boundary between instructions and data is drawn. Everything
retrieved from a document is untrusted: a PDF can contain a sentence addressed
to a language model, and that sentence reaches this point looking exactly like
any other passage. The defence is structural rather than a filter — retrieved
text is placed inside marked source material that the template's instructions
describe as data to be quoted, never as instructions to be followed. No attempt
is made to detect malicious wording, because a filter can be rephrased around,
whereas a boundary holds regardless of what the text says.

One thing the boundary does need defending against is a document forging the
boundary itself. A passage containing the source-material marker could appear
to close the quoted region and continue as though it were application text, so
markers occurring in retrieved content are neutralised on the way in.

Templates are files with a name and a version, and every context records which
one produced it. A template is resolved by identifier against this package's
own directory — never by a path from a caller — so prompt selection cannot be
turned into reading an arbitrary file.

This component never calls a language model. It returns the assembled context
and stops.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from importlib import resources
from string import Template

from rag.domain.models import PromptContext, PromptMetadata, RerankedChunk
from rag.retrieval.interfaces import BasePromptAugmenter

#: Package holding the versioned prompt templates.
_PROMPT_PACKAGE = "rag.retrieval.prompts"

#: A template's filename is derived from its name and version alone.
_TEMPLATE_FILENAME = "{name}_{version}.txt"

#: Permitted characters in a template name or version. Anything outside this
#: set — separators, dots, anything that could traverse a path — is refused
#: before a filename is built.
_SAFE_IDENTIFIER = re.compile(r"^[a-z0-9_]+$")

#: Markers delimiting untrusted content in the rendered prompt.
_BEGIN_MARKER = "===== BEGIN SOURCE MATERIAL ====="
_END_MARKER = "===== END SOURCE MATERIAL ====="

#: Any line a document could use to imitate a marker. Matched loosely, on the
#: marker's distinctive wording rather than its exact punctuation, so that a
#: near-miss forgery is neutralised too.
_MARKER_FORGERY = re.compile(
    r"=*\s*(?:BEGIN|END)\s+SOURCE\s+MATERIAL\s*=*", re.IGNORECASE
)

#: What a neutralised marker becomes: visible to a reader, inert as a boundary.
_NEUTRALISED = "[marker removed]"


class PromptTemplateNotFoundError(RuntimeError):
    """Raised when a prompt template cannot be resolved.

    Names the template and version requested, never a filesystem path.
    """


class TemplatePromptAugmenter(BasePromptAugmenter):
    """Renders retrieved chunks into a versioned prompt context.

    Attributes:
        prompt_name: Logical name of the template in use.
        prompt_version: Version of the template in use.
    """

    def __init__(
        self, *, prompt_name: str = "retrieval_context", prompt_version: str = "v1"
    ) -> None:
        """Load the identified template.

        Args:
            prompt_name: Logical name of the template.
            prompt_version: Version identifier of the template.

        Raises:
            ValueError: If the name or version contains anything outside
                lowercase letters, digits, and underscores.
            PromptTemplateNotFoundError: If no such template exists.
        """
        _validate_identifier(prompt_name, "prompt_name")
        _validate_identifier(prompt_version, "prompt_version")

        self.prompt_name = prompt_name
        self.prompt_version = prompt_version
        self._template = Template(_read_template(prompt_name, prompt_version))

    @property
    def metadata(self) -> PromptMetadata:
        """Identity of the template this augmenter renders.

        Returns:
            The template's name and version.
        """
        return PromptMetadata(
            prompt_name=self.prompt_name, prompt_version=self.prompt_version
        )

    def augment(self, query: str, chunks: Sequence[RerankedChunk]) -> PromptContext:
        """Build the context block for a query.

        Args:
            query: The user's query.
            chunks: The chunks to include, most relevant first.

        Returns:
            The rendered context together with the template's identity.
        """
        source_material = (
            "\n\n".join(
                _render_passage(position, chunk)
                for position, chunk in enumerate(chunks, start=1)
            )
            or "(no relevant source material was found)"
        )

        rendered = self._template.substitute(
            source_material=source_material,
            question=neutralise_markers(query),
        )
        return PromptContext(
            chunks=tuple(chunks),
            rendered_text=rendered,
            prompt_metadata=self.metadata,
        )


def neutralise_markers(text: str) -> str:
    """Disarm any source-material marker appearing in untrusted text.

    Args:
        text: Untrusted text, from a document or from a caller's query.

    Returns:
        The text with marker-like sequences replaced, so that it cannot appear
        to close the quoted region and continue as application instructions.
    """
    return _MARKER_FORGERY.sub(_NEUTRALISED, text)


def _render_passage(position: int, chunk: RerankedChunk) -> str:
    """Render one retrieved passage with its provenance.

    Provenance is written by the pipeline, not taken from the document, so a
    citation cannot be forged by a document claiming a different source.

    Args:
        position: The passage's citation number.
        chunk: The reranked chunk to render.

    Returns:
        The passage as it appears in the source material.
    """
    source = chunk.chunk
    location = f"document={source.document_id} page={source.page_number}"
    if source.section:
        location = f"{location} section={neutralise_markers(source.section)}"

    return f"[{position}] ({location})\n{neutralise_markers(source.text)}"


def _validate_identifier(value: str, field: str) -> None:
    """Check that a template identifier is safe to build a filename from.

    Args:
        value: The identifier supplied.
        field: Which field is being checked, for the error message.

    Raises:
        ValueError: If the identifier contains anything unexpected.
    """
    if not _SAFE_IDENTIFIER.match(value):
        raise ValueError(
            f"{field} may contain only lowercase letters, digits, and underscores."
        )


def _read_template(name: str, version: str) -> str:
    """Read a template from the package's own prompt directory.

    Args:
        name: Logical name of the template.
        version: Version identifier of the template.

    Returns:
        The template's text.

    Raises:
        PromptTemplateNotFoundError: If no such template exists.
    """
    filename = _TEMPLATE_FILENAME.format(name=name, version=version)
    resource = resources.files(_PROMPT_PACKAGE).joinpath(filename)
    try:
        return resource.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError) as exc:
        raise PromptTemplateNotFoundError(
            f"No prompt template named {name!r} at version {version!r}."
        ) from exc
