from agent_b_synth.nodes.node_chapterize import normalize_chapters


def test_normalize_chapters_adds_emotion_tags():
    """Test that emotion tags are added to chapters that don't have them."""
    input_chapters = [
        "Hello",
        "Welcome to this guided meditation",
        "Imagine yourself standing in the city"
    ]

    result = normalize_chapters(input_chapters)

    assert len(result) == 3
    assert result[0].startswith("<emotion value=\"serene\" />")
    assert result[1].startswith("<emotion value=\"serene\" />")
    assert result[2].startswith("<emotion value=\"serene\" />")


def test_normalize_chapters_merges_standalone_breaks():
    """Test that standalone break tags are merged with preceding chapter."""
    input_chapters = [
        "Hello",
        "<break time=\"1.5s\" />",
        "Welcome to this guided meditation"
    ]

    result = normalize_chapters(input_chapters)

    assert len(result) == 2
    assert "<break time=\"1.5s\" />" in result[0]
    assert result[1].startswith("<emotion value=\"serene\" />")


def test_normalize_chapters_preserves_existing_emotion_tags():
    """Test that chapters with existing emotion tags are not modified."""
    input_chapters = [
        "<emotion value=\"serene\" />Hello",
        "<emotion value=\"serene\" />Welcome"
    ]

    result = normalize_chapters(input_chapters)

    assert len(result) == 2
    assert result[0].startswith("<emotion value=\"serene\" />")
    assert result[1].startswith("<emotion value=\"serene\" />")


def test_normalize_chapters_ensures_break_at_end():
    """Test that every chapter ends with a break tag."""
    input_chapters = [
        "Hello",
        "Welcome to this guided meditation",
        "Imagine yourself standing in the city"
    ]

    result = normalize_chapters(input_chapters)

    assert len(result) == 3
    for chapter in result:
        assert chapter.strip().endswith("<break time=\"1.5s\" />")


def test_normalize_chapters_empty_input():
    """Test that empty input returns empty list."""
    result = normalize_chapters([])
    assert result == []


def test_normalize_chapters_single_chapter():
    """Test that single chapter is handled correctly."""
    input_chapters = ["Hello"]

    result = normalize_chapters(input_chapters)

    assert len(result) == 1
    assert result[0].startswith("<emotion value=\"serene\" />")
    assert result[0].strip().endswith("<break time=\"1.5s\" />")


def test_normalize_chapters_with_existing_break():
    """Test that chapters with existing break tags are not duplicated."""
    input_chapters = [
        "Hello<break time=\"2s\" />",
        "Welcome"
    ]

    result = normalize_chapters(input_chapters)

    assert len(result) == 2
    assert result[0].strip().endswith("<break time=\"2s\" />")
    assert result[1].startswith("<emotion value=\"serene\" />")
    assert result[1].strip().endswith("<break time=\"1.5s\" />")
