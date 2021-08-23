import pytest

from pyconversations.message import RedditPost


@pytest.fixture
def null_post():
    """Returns a RedditPost with a UID of 0"""
    return RedditPost(uid=0)


@pytest.fixture
def mock_json_post():
    """Returns a mock post of a cached RedditPost"""
    return {
        'uid': 1234,
        'text': 'u/userA This is a Reddit post!',
        'author': 'poster',
        'created_at': 9999999.0,
        'reply_to':   [1233],
        'platform':   'Reddit',
        'tags':       ['test_tag'],
        'lang':       'en'
    }


@pytest.fixture
def mock_post(mock_json_post):
    """Returns a mock post object"""
    return RedditPost(**mock_json_post)


@pytest.fixture
def mock_raw_post():
    test_keys = ["id", "author_name", 'created_utc', "parent_id", "link_id", "score", "body", 'subreddit']
    test_vals = ['c1234', 'poster', '9999999999', 't3_c1230', '', '4', 'text here please', 'r/reddit.com']
    return {
        k: v for k, v in zip(test_keys, test_vals)
    }


def test_null_post(null_post):
    assert null_post.uid == 0
    assert null_post.uid == hash(null_post)
    assert null_post.text == ''
    assert null_post.author is None
    assert null_post.created_at is None
    assert null_post.reply_to == set()
    assert null_post.platform == 'Reddit'
    assert null_post.tags == set()
    assert null_post.lang is None


def test_update_post(null_post, mock_json_post):
    null_post.text = mock_json_post['text']
    assert null_post.text == mock_json_post['text']

    null_post.author = mock_json_post['author']
    assert null_post.author == mock_json_post['author']

    null_post.created_at = mock_json_post['created_at']
    assert null_post.created_at.timestamp() == mock_json_post['created_at']

    for uid in mock_json_post['reply_to']:
        null_post.add_reply_to(uid)
    assert null_post.reply_to == set(mock_json_post['reply_to'])

    for uid in mock_json_post['reply_to']:
        null_post.remove_reply_to(uid)
    assert null_post.reply_to == set()

    null_post.platform = 'Reddit'
    assert null_post.platform == 'Reddit'

    for tag in mock_json_post['tags']:
        null_post.add_tag(tag)
    assert null_post.tags == set(mock_json_post['tags'])

    for tag in mock_json_post['tags']:
        null_post.remove_tag(tag)
    assert null_post.tags == set()

    null_post.lang = mock_json_post['lang']
    assert null_post.lang == mock_json_post['lang']


def test_post_from_to_json(mock_json_post):
    t = RedditPost.from_json(mock_json_post)

    # revert internal change of JSON
    mock_json_post['created_at'] = mock_json_post['created_at'].timestamp()

    out = t.to_json()

    for k, v in mock_json_post.items():
        assert v == out[k]

    for k, v in out.items():
        assert v == mock_json_post[k]


def test_to_from_json_null_datetime(mock_json_post):
    mock_json_post['created_at'] = None
    post = RedditPost.from_json(mock_json_post)
    out = post.to_json()

    for k, v in mock_json_post.items():
        assert v == out[k]

    for k, v in out.items():
        assert v == mock_json_post[k]


def test_post_datetime_parsing():
    from datetime import datetime

    dt = '9999999999.0'
    assert datetime(2286, 11, 20, 12, 46, 39) == RedditPost.parse_datestr(dt)


def test_reddit_mentions(mock_post):
    assert mock_post.get_mentions() == {'userA', 'poster'}


def test_read_raw_post(mock_raw_post):
    from datetime import datetime

    post = RedditPost.parse_raw(mock_raw_post)[0]
    assert post.uid == "c1234"
    assert post.text == "text here please"
    assert post.author == 'poster'
    assert post.created_at == datetime(2286, 11, 20, 12, 46, 39)
    assert post.reply_to == {'c1230'}
    assert post.platform == 'Reddit'
    assert post.tags == set()
    assert post.lang is None

    mock_raw_post['created'] = mock_raw_post['created_utc']
    del mock_raw_post['created_utc']
    post = RedditPost.parse_raw(mock_raw_post)[0]
    assert post.created_at == datetime(2286, 11, 20, 12, 46, 39)

    mock_raw_post['title'] = 'title:'
    post = RedditPost.parse_raw(mock_raw_post)[0]
    assert post.text == "title:\ntext here please"

    mock_raw_post['errorr'] = 'error'
    with pytest.raises(KeyError):
        RedditPost.parse_raw(mock_raw_post)
