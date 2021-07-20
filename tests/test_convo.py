import pytest

from pyconversations.convo import Conversation
from pyconversations.message import Tweet


@pytest.fixture
def mock_tweet():
    return Tweet(
        uid=1,
        text='test text',
        author=1,
        reply_to={0}
    )


@pytest.fixture
def mock_root_tweet():
    return Tweet(
        uid=0,
        author=0,
        text='Root tweet text',
    )


@pytest.fixture
def mock_convo(mock_root_tweet):
    convo = Conversation()
    convo.add_post(mock_root_tweet)
    return convo


@pytest.fixture
def mock_convo_path(mock_root_tweet, mock_tweet):
    convo = Conversation()
    convo.add_post(mock_root_tweet)
    convo.add_post(mock_tweet)
    return convo


@pytest.fixture
def mock_multi_convo():
    convo = Conversation()
    for ix in range(10):
        reps = set()
        if ix > 1:
            reps.add(ix - 2)
        convo.add_post(Tweet(uid=ix, reply_to=reps))

    return convo


def test_build_conversation(mock_tweet):
    conversation = Conversation()

    uid = mock_tweet.uid

    conversation.add_post(mock_tweet)
    assert uid in conversation.posts

    conversation.remove_post(uid)
    assert uid not in conversation.posts

    with pytest.raises(KeyError):
        conversation.remove_post(uid)


def test_add_convo_to_self(mock_tweet):
    conversation = Conversation()

    uid = mock_tweet.uid

    conversation.add_post(mock_tweet)
    conversation = conversation + conversation

    assert uid in conversation.posts


def test_add_convo(mock_tweet, mock_root_tweet):
    root_convo = Conversation()
    root_convo.add_post(mock_root_tweet)

    convo = Conversation()
    convo.add_post(mock_tweet)

    full = root_convo + convo

    assert 0 in full.posts
    assert 1 in full.posts


def test_convo_segmentation(mock_multi_convo):
    segs = mock_multi_convo.segment()
    assert len(segs) == 2

    even, odd = segs
    for ix in range(5):
        assert 2 * ix in even.posts
        assert (2 * ix) + 1 in odd.posts


def test_to_from_json(mock_convo):
    assert 0 in mock_convo.posts

    raw_json = mock_convo.to_json()
    new_convo = Conversation.from_json(raw_json)

    assert 0 in new_convo.posts


def test_stats(mock_convo):
    assert mock_convo.sources == {0}
    assert mock_convo.sources == {0}


def test_stats_path(mock_convo_path):
    assert mock_convo_path.sources == {0}
    assert mock_convo_path.text_stream == ['Root tweet text', 'test text']


def test_stats_no_parent(mock_tweet):
    convo = Conversation()
    convo.add_post(mock_tweet)

    assert convo.sources == {1}


def test_conversation_filter_min_char(mock_convo_path):
    assert len(mock_convo_path.posts) == 2
    mock_convo_path.posts[0].text = ''

    filt = mock_convo_path.filter(min_chars=1)
    assert len(filt.posts) == 1


def test_conversation_filter_by_langs(mock_convo_path):
    assert len(mock_convo_path.posts) == 2
    filt = mock_convo_path.filter(by_langs={'en'})
    assert len(filt.posts) == 0


def test_conversation_filter_by_tags(mock_convo_path):
    assert len(mock_convo_path.posts) == 2
    filt = mock_convo_path.filter(by_tags={'#FakeNews'})
    assert len(filt.posts) == 0


def test_conversation_filter_by_before(mock_convo_path):
    from datetime import datetime

    assert len(mock_convo_path.posts) == 2
    filt = mock_convo_path.filter(before=datetime(2020, 12, 1, 11, 11, 11))
    assert len(filt.posts) == 0


def test_conversation_filter_by_after(mock_convo_path):
    from datetime import datetime

    assert len(mock_convo_path.posts) == 2
    filt = mock_convo_path.filter(after=datetime(2020, 12, 1, 11, 11, 11))
    assert len(filt.posts) == 0


@pytest.fixture
def mock_temporal_convo():
    from datetime import datetime

    conv = Conversation()
    conv.add_post(Tweet(uid=0, text='@tweet 0', created_at=datetime(2020, 12, 1, 10, 5, 5)))
    conv.add_post(Tweet(uid=1, text='@tweet 1', created_at=datetime(2020, 12, 1, 10, 5, 35), reply_to={0}))
    conv.add_post(Tweet(uid=2, text='@tweet 2', created_at=datetime(2020, 12, 1, 10, 5, 45), reply_to={0}))
    conv.add_post(Tweet(uid=3, text='@tweet 3', created_at=datetime(2020, 12, 1, 12, 5, 45), reply_to={1}))
    return conv


def test_ordered_properties(mock_temporal_convo):
    assert mock_temporal_convo.time_order == list(range(4))
    assert mock_temporal_convo.text_stream == [f'@tweet {i}' for i in range(4)]


def test_convo_redaction(mock_temporal_convo):
    mock_temporal_convo.redact()
    assert mock_temporal_convo.text_stream == [f'@USER0 {i}' for i in range(4)]


def test_conversation_post_merge(mock_root_tweet, mock_tweet):
    convo = Conversation()
    convo.add_post(mock_root_tweet)
    m = Tweet(uid=0, text=mock_tweet.text)
    convo.add_post(m)

    assert convo.posts[0].text == 'Root tweet text'


def test_conversation_post_merge_text():
    t0 = Tweet(uid=0, text='test this')
    t1 = Tweet(uid=0, text='longer text')
    convo = Conversation()
    convo.add_post(t0)
    convo.add_post(t1)
    assert len(convo.posts) == 1
    assert convo.posts[0].text == 'longer text'


def test_conversation_post_merge_created_at():
    from datetime import datetime

    convo = Conversation()

    convo.add_post(Tweet(uid=0, created_at=datetime(2020, 12, 1, 12, 12, 19)))
    convo.add_post(Tweet(uid=0, created_at=datetime(2020, 12, 1, 12, 12, 12)))

    convo.add_post(Tweet(uid=1, created_at=datetime(2020, 12, 1, 12, 12, 12)))
    convo.add_post(Tweet(uid=1, created_at=datetime(2020, 12, 1, 12, 12, 19)))

    assert len(convo.posts) == 2
    assert convo.posts[0].created_at == datetime(2020, 12, 1, 12, 12, 12)
    assert convo.posts[1].created_at == datetime(2020, 12, 1, 12, 12, 12)


def test_conversation_post_merge_lang():
    convo = Conversation()

    convo.add_post(Tweet(uid=0, lang=None))
    convo.add_post(Tweet(uid=0, lang='en'))
    convo.add_post(Tweet(uid=0, lang='en'))

    assert len(convo.posts) == 1
    assert convo.posts[0].lang == 'en'


def test_get_depth(mock_convo):
    assert mock_convo.get_depth(0) == 0
    assert mock_convo.get_depth(0) == 0

    assert mock_convo.depths == [0]
    assert mock_convo.depths == [0]

    assert mock_convo.tree_depth == 0
    assert mock_convo.tree_depth == 0

    assert mock_convo.widths == [1]
    assert mock_convo.widths == [1]

    assert mock_convo.tree_width == 1
    assert mock_convo.tree_width == 1


def test_path_shape(mock_convo_path):
    assert mock_convo_path.get_depth(1) == 1
    assert mock_convo_path.get_depth(1) == 1
