"""Unit tests for tree_utils.py pure transformation functions.

These tests use hand-crafted dicts — no browser or HTTP server needed.
Tests are organized by function, covering core behavior, edge cases,
and the interaction between compaction rules.
"""

from __future__ import annotations

import copy

from selenium_browser_automation.tree_utils import (
    compact_aria_tree,
    compact_visual_tree,
    serialize_aria_snapshot,
    serialize_visual_tree,
)

# -- compact_aria_tree ---------------------------------------------------------


class TestCompactAriaTreeNoneAndLeafNodes:
    """None input, text nodes, and simple leaf elements."""

    def test_none_returns_none(self) -> None:
        assert compact_aria_tree(None) is None

    def test_text_node_passes_through(self) -> None:
        node = {'type': 'text', 'content': 'Hello'}
        assert compact_aria_tree(node) == node

    def test_semantic_leaf_preserved(self) -> None:
        """Non-generic leaf nodes are always preserved."""
        node = {'role': 'button', 'name': 'Submit'}
        assert compact_aria_tree(node) == node

    def test_named_generic_preserved(self) -> None:
        """Generic with a name is semantically meaningful, keep it."""
        node = {'role': 'generic', 'name': 'Container'}
        assert compact_aria_tree(node) == {'role': 'generic', 'name': 'Container'}


class TestCompactAriaTreeRule1EmptyGenerics:
    """Rule 1: Remove empty generics (no name, description, or children)."""

    def test_empty_generic_removed(self) -> None:
        node = {'role': 'generic'}
        assert compact_aria_tree(node) is None

    def test_empty_generic_explicit_empty_name(self) -> None:
        node = {'role': 'generic', 'name': ''}
        assert compact_aria_tree(node) is None

    def test_generic_with_description_preserved(self) -> None:
        node = {'role': 'generic', 'description': 'Important section'}
        result = compact_aria_tree(node)
        assert result is not None
        assert result.get('description') == 'Important section'

    def test_generic_with_hidden_marker_preserved(self) -> None:
        """Empty generic with visibility marker kept for debugging value."""
        node = {'role': 'generic', 'hidden': 'aria-hidden'}
        result = compact_aria_tree(node)
        assert result is not None
        assert result.get('hidden') == 'aria-hidden'

    def test_generic_with_visually_hidden_marker_preserved(self) -> None:
        node = {'role': 'generic', 'visuallyHidden': 'clipped'}
        result = compact_aria_tree(node)
        assert result is not None


class TestCompactAriaTreeRule2SingleChildCollapse:
    """Rule 2: Collapse single-child generic chains (unwrap wrapper divs)."""

    def test_single_child_generic_collapsed(self) -> None:
        """<div><button>Click</button></div> → button promoted."""
        node = {
            'role': 'generic',
            'children': [{'role': 'button', 'name': 'Click'}],
        }
        result = compact_aria_tree(node)
        assert result == {'role': 'button', 'name': 'Click'}

    def test_nested_wrapper_chain_collapsed(self) -> None:
        """<div><div><div><button>Deep</button></div></div></div> → button."""
        node = {
            'role': 'generic',
            'children': [
                {
                    'role': 'generic',
                    'children': [
                        {
                            'role': 'generic',
                            'children': [{'role': 'button', 'name': 'Deep'}],
                        },
                    ],
                },
            ],
        }
        result = compact_aria_tree(node)
        assert result == {'role': 'button', 'name': 'Deep'}

    def test_named_generic_not_collapsed(self) -> None:
        """Generic with a name is meaningful — don't collapse."""
        node = {
            'role': 'generic',
            'name': 'Wrapper',
            'children': [{'role': 'button', 'name': 'Click'}],
        }
        result = compact_aria_tree(node)
        assert result is not None
        assert result.get('role') == 'generic'
        assert result.get('name') == 'Wrapper'

    def test_hidden_generic_not_collapsed(self) -> None:
        """Generic with hidden marker — preserve structure for debugging."""
        node = {
            'role': 'generic',
            'hidden': 'aria-hidden',
            'children': [{'role': 'button', 'name': 'Click'}],
        }
        result = compact_aria_tree(node)
        assert result is not None
        assert result.get('role') == 'generic'
        assert result.get('hidden') == 'aria-hidden'

    def test_multi_child_generic_not_collapsed(self) -> None:
        """Generic with multiple children kept as container."""
        node = {
            'role': 'generic',
            'children': [
                {'role': 'button', 'name': 'A'},
                {'role': 'button', 'name': 'B'},
            ],
        }
        result = compact_aria_tree(node)
        assert result is not None
        assert len(result.get('children', [])) == 2


class TestCompactAriaTreeRule3RedundantText:
    """Rule 3: Remove redundant text children when name matches concatenation."""

    def test_single_redundant_text_removed(self) -> None:
        """<button>Submit</button> where name='Submit' and child is text 'Submit'."""
        node = {
            'role': 'button',
            'name': 'Submit',
            'children': [{'type': 'text', 'content': 'Submit'}],
        }
        result = compact_aria_tree(node)
        assert result == {'role': 'button', 'name': 'Submit'}

    def test_multi_text_concatenation_matches(self) -> None:
        """Multiple text children that concatenate to match name."""
        node = {
            'role': 'link',
            'name': 'Click here',
            'children': [
                {'type': 'text', 'content': 'Click'},
                {'type': 'text', 'content': 'here'},
            ],
        }
        result = compact_aria_tree(node)
        assert result == {'role': 'link', 'name': 'Click here'}

    def test_non_matching_text_preserved(self) -> None:
        """Text children that don't match name are kept."""
        node = {
            'role': 'button',
            'name': 'Submit',
            'children': [{'type': 'text', 'content': 'Different'}],
        }
        result = compact_aria_tree(node)
        assert result is not None
        assert len(result.get('children', [])) == 1

    def test_mixed_children_not_collapsed(self) -> None:
        """Mix of text and element children — rule 3 doesn't apply."""
        node = {
            'role': 'link',
            'name': 'Go',
            'children': [
                {'type': 'text', 'content': 'Go'},
                {'role': 'img', 'name': 'arrow'},
            ],
        }
        result = compact_aria_tree(node)
        assert result is not None
        # img child survives, text child might be removed but img blocks rule 3
        assert any(c.get('role') == 'img' for c in result.get('children', []))

    def test_unicode_nfkc_normalization(self) -> None:
        """Ellipsis (…) normalizes to three dots (...) via NFKC."""
        node = {
            'role': 'button',
            'name': 'More\u2026',  # name has ellipsis
            'children': [{'type': 'text', 'content': 'More...'}],  # text has dots
        }
        result = compact_aria_tree(node)
        assert result is not None
        # NFKC normalizes ellipsis → ..., so they match → text removed
        assert 'children' not in result


class TestCompactAriaTreeDoesNotMutate:
    """Verify compaction returns new nodes, not mutated originals."""

    def test_original_node_unchanged(self) -> None:
        original = {
            'role': 'generic',
            'children': [
                {'role': 'generic'},  # will be removed (empty)
                {'role': 'button', 'name': 'Keep'},
            ],
        }
        frozen = copy.deepcopy(original)
        compact_aria_tree(original)
        assert original == frozen


# -- compact_visual_tree -------------------------------------------------------


class TestCompactVisualTree:
    """Visual tree compaction — same rules as ARIA but without visuallyHidden."""

    def test_text_node_passes_through(self) -> None:
        node = {'type': 'text', 'content': 'Hello'}
        assert compact_visual_tree(node) == node

    def test_empty_generic_removed(self) -> None:
        assert compact_visual_tree({'role': 'generic'}) is None

    def test_single_child_generic_collapsed(self) -> None:
        node = {
            'role': 'generic',
            'children': [{'role': 'button', 'name': 'Click'}],
        }
        assert compact_visual_tree(node) == {'role': 'button', 'name': 'Click'}

    def test_hidden_marker_preserves_empty_generic(self) -> None:
        """Visual tree uses 'hidden' but not 'visuallyHidden'."""
        node = {'role': 'generic', 'hidden': 'css'}
        result = compact_visual_tree(node)
        assert result is not None

    def test_redundant_text_removed(self) -> None:
        node = {
            'role': 'button',
            'name': 'OK',
            'children': [{'type': 'text', 'content': 'OK'}],
        }
        assert compact_visual_tree(node) == {'role': 'button', 'name': 'OK'}


# -- serialize_aria_snapshot ---------------------------------------------------


class TestSerializeAriaSnapshot:
    """ARIA snapshot serialization to Playwright-format YAML."""

    def test_none_returns_empty(self) -> None:
        assert serialize_aria_snapshot(None) == ''

    def test_simple_button(self) -> None:
        assert serialize_aria_snapshot({'role': 'button', 'name': 'Go'}) == '- button "Go"'

    def test_unnamed_generic(self) -> None:
        assert serialize_aria_snapshot({'role': 'generic'}) == '- generic'

    def test_heading_with_level(self) -> None:
        node = {'role': 'heading', 'name': 'Title', 'level': 2}
        assert serialize_aria_snapshot(node) == '- heading "Title" [level=2]'

    def test_checkbox_checked(self) -> None:
        node = {'role': 'checkbox', 'name': 'Agree', 'checked': True}
        assert serialize_aria_snapshot(node) == '- checkbox "Agree" [checked]'

    def test_checkbox_unchecked(self) -> None:
        node = {'role': 'checkbox', 'name': 'Agree', 'checked': False}
        assert serialize_aria_snapshot(node) == '- checkbox "Agree" [unchecked]'

    def test_checkbox_mixed(self) -> None:
        node = {'role': 'checkbox', 'name': 'Partial', 'checked': 'mixed'}
        assert serialize_aria_snapshot(node) == '- checkbox "Partial" [checked=mixed]'

    def test_selected_true(self) -> None:
        node = {'role': 'tab', 'name': 'Home', 'selected': True}
        assert serialize_aria_snapshot(node) == '- tab "Home" [selected]'

    def test_selected_false(self) -> None:
        node = {'role': 'tab', 'name': 'About', 'selected': False}
        assert serialize_aria_snapshot(node) == '- tab "About" [selected=false]'

    def test_pressed_true(self) -> None:
        node = {'role': 'button', 'name': 'Bold', 'pressed': True}
        assert serialize_aria_snapshot(node) == '- button "Bold" [pressed]'

    def test_pressed_false(self) -> None:
        node = {'role': 'button', 'name': 'Bold', 'pressed': False}
        assert serialize_aria_snapshot(node) == '- button "Bold" [pressed=false]'

    def test_pressed_mixed(self) -> None:
        node = {'role': 'button', 'name': 'Toggle', 'pressed': 'mixed'}
        assert serialize_aria_snapshot(node) == '- button "Toggle" [pressed=mixed]'

    def test_expanded_true(self) -> None:
        node = {'role': 'button', 'name': 'Menu', 'expanded': True}
        assert serialize_aria_snapshot(node) == '- button "Menu" [expanded]'

    def test_expanded_false(self) -> None:
        node = {'role': 'button', 'name': 'Menu', 'expanded': False}
        assert serialize_aria_snapshot(node) == '- button "Menu" [expanded=false]'

    def test_disabled(self) -> None:
        node = {'role': 'button', 'name': 'Send', 'disabled': True}
        assert serialize_aria_snapshot(node) == '- button "Send" [disabled]'

    def test_url_attribute(self) -> None:
        node = {'role': 'link', 'name': 'Home', 'url': '/home'}
        assert serialize_aria_snapshot(node) == '- link "Home" [url=/home]'

    def test_hidden_marker(self) -> None:
        node = {'role': 'button', 'name': 'X', 'hidden': 'aria-hidden'}
        assert serialize_aria_snapshot(node) == '- button "X" [hidden:aria-hidden]'

    def test_visually_hidden_marker(self) -> None:
        node = {'role': 'generic', 'name': 'SR', 'visuallyHidden': 'clipped'}
        assert serialize_aria_snapshot(node) == '- generic "SR" [visually-hidden:clipped]'

    def test_multiple_attributes(self) -> None:
        """Multiple attributes appear in bracket-separated format."""
        node = {'role': 'checkbox', 'name': 'Terms', 'checked': True, 'disabled': True}
        output = serialize_aria_snapshot(node)
        assert 'checked' in output
        assert 'disabled' in output

    def test_text_node(self) -> None:
        node = {'type': 'text', 'content': 'Hello world'}
        assert serialize_aria_snapshot(node) == '- text: Hello world'

    def test_text_node_whitespace_normalized(self) -> None:
        """Text node content gets whitespace-collapsed."""
        node = {'type': 'text', 'content': '  Hello   world  '}
        assert serialize_aria_snapshot(node) == '- text: Hello world'

    def test_children_indented(self) -> None:
        node = {
            'role': 'list',
            'name': 'Items',
            'children': [
                {'role': 'listitem', 'name': 'First'},
                {'role': 'listitem', 'name': 'Second'},
            ],
        }
        output = serialize_aria_snapshot(node)
        assert '- list "Items":' in output
        assert '  - listitem "First"' in output
        assert '  - listitem "Second"' in output

    def test_nested_indentation(self) -> None:
        """Three levels of nesting get progressively indented."""
        node = {
            'role': 'navigation',
            'name': 'Nav',
            'children': [
                {
                    'role': 'list',
                    'children': [
                        {'role': 'listitem', 'name': 'Home'},
                    ],
                },
            ],
        }
        output = serialize_aria_snapshot(node)
        lines = output.split('\n')
        assert lines[0] == '- navigation "Nav":'
        assert lines[1] == '  - list:'
        assert lines[2] == '    - listitem "Home"'

    def test_name_with_quotes_escaped(self) -> None:
        node = {'role': 'button', 'name': 'Say "hello"'}
        output = serialize_aria_snapshot(node)
        assert r'\"hello\"' in output


# -- serialize_visual_tree -----------------------------------------------------


class TestSerializeVisualTree:
    """Visual tree serialization — same YAML format, slightly different attrs."""

    def test_none_returns_empty(self) -> None:
        assert serialize_visual_tree(None) == ''

    def test_simple_button(self) -> None:
        assert serialize_visual_tree({'role': 'button', 'name': 'Go'}) == '- button "Go"'

    def test_text_node(self) -> None:
        node = {'type': 'text', 'content': 'Hello'}
        assert serialize_visual_tree(node) == '- text: Hello'

    def test_hidden_marker(self) -> None:
        node = {'role': 'generic', 'name': 'X', 'hidden': 'css'}
        output = serialize_visual_tree(node)
        assert '[hidden:css]' in output

    def test_level_attribute(self) -> None:
        node = {'role': 'heading', 'name': 'Title', 'level': 1}
        output = serialize_visual_tree(node)
        assert '[level=1]' in output

    def test_children_indented(self) -> None:
        node = {
            'role': 'list',
            'name': 'Items',
            'children': [
                {'role': 'listitem', 'name': 'A'},
            ],
        }
        output = serialize_visual_tree(node)
        assert '- list "Items":' in output
        assert '  - listitem "A"' in output

    def test_checkbox_states(self) -> None:
        """Visual tree serializes checkbox states same as ARIA."""
        assert '[checked]' in serialize_visual_tree({'role': 'checkbox', 'name': 'X', 'checked': True})
        assert '[unchecked]' in serialize_visual_tree({'role': 'checkbox', 'name': 'X', 'checked': False})
        assert '[checked=mixed]' in serialize_visual_tree({'role': 'checkbox', 'name': 'X', 'checked': 'mixed'})

    def test_no_visually_hidden_attr(self) -> None:
        """Visual tree doesn't have visuallyHidden — only ARIA does."""
        node = {'role': 'generic', 'name': 'X', 'visuallyHidden': 'clipped'}
        output = serialize_visual_tree(node)
        # visuallyHidden is not handled by visual serializer, should not appear
        assert 'visually-hidden' not in output
