import pytest

from pyconversations.message import ChanPost


@pytest.fixture
def null_post():
    """Returns a ChanPost with a UID of 0"""
    return ChanPost(uid=0)


@pytest.fixture
def mock_json_post():
    """Returns a mock post of a cached ChanPost"""
    return {
        'uid': 1234,
        'text': 'This is a 4chan post!',
        'author': 'poster',
        'created_at': 9999999.0,
        'reply_to':   [1233],
        'platform':   '4chan',
        'tags':       ['test_tag'],
        'lang':       'en'
    }


@pytest.fixture
def mock_raw_post():
    return {
        "bumplimit": 0,
        "com": "raw comment >>12345",
        "custom_spoiler": 1,
        "ext": ".jpg",
        "filename": "test",
        "fsize": 0,
        "h": 0,
        "imagelimit": 0,
        "images": 0,
        "last_modified": 9999999999,
        "last_replies": [],
        "md5": "",
        "name": "Anonymous",
        "no": 12360,
        "now": "",
        "replies": 0,
        "resto": 0,
        "semantic_url": "subject",
        "sub": "subject",
        "tim": 9999999999315,
        "time": 9999999999,
        "tn_h": 0,
        "tn_w": 0,
        "w": 0
    }


def test_null_post(null_post):
    assert null_post.uid == 0
    assert null_post.uid == hash(null_post)
    assert null_post.text == ''
    assert null_post.author is None
    assert null_post.created_at is None
    assert null_post.reply_to == set()
    assert null_post.platform == '4chan'
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

    null_post.platform = '4chan'
    assert null_post.platform == '4chan'

    for tag in mock_json_post['tags']:
        null_post.add_tag(tag)
    assert null_post.tags == set(mock_json_post['tags'])

    for tag in mock_json_post['tags']:
        null_post.remove_tag(tag)
    assert null_post.tags == set()

    null_post.lang = mock_json_post['lang']
    assert null_post.lang == mock_json_post['lang']


def test_post_from_to_json(mock_json_post):
    t = ChanPost.from_json(mock_json_post)

    # revert internal change of JSON
    mock_json_post['created_at'] = mock_json_post['created_at'].timestamp()

    out = t.to_json()
    for k, v in mock_json_post.items():
        assert v == out[k]

    for k, v in out.items():
        assert v == mock_json_post[k]


def test_to_from_json_null_datetime(mock_json_post):
    mock_json_post['created_at'] = None
    post = ChanPost.from_json(mock_json_post)
    out = post.to_json()

    for k, v in mock_json_post.items():
        assert v == out[k]

    for k, v in out.items():
        assert v == mock_json_post[k]


def test_post_datetime_parsing():
    from datetime import datetime

    dt = '9999999999.0'
    assert datetime(2286, 11, 20, 12, 46, 39) == ChanPost.parse_datestr(dt)


def test_read_raw_post(mock_raw_post):
    from datetime import datetime

    post = ChanPost.parse_raw(mock_raw_post)
    assert post.uid == 12360
    assert post.text == "raw comment >>12345"
    assert post.author == 'Anonymous'
    assert post.created_at == datetime(2286, 11, 20, 12, 46, 39)
    assert post.reply_to == {12345}
    assert post.platform == '4chan'
    assert post.tags == set()
    assert post.lang is None

    mock_raw_post['com'] += ' >>12360'
    post = ChanPost.parse_raw(mock_raw_post)
    assert post.reply_to == {12345}
    assert post.text == "raw comment >>12345 >>12360"

    del mock_raw_post['com']
    assert ChanPost.parse_raw(mock_raw_post) is None
