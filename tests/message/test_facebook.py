import pytest

from pyconversations.message import FBPost


@pytest.fixture
def null_post():
    """Returns a FBPost with a UID of 0"""
    return FBPost(uid=0)


@pytest.fixture
def mock_json_post():
    """Returns a mock post of a cached FBPost"""
    return {
        'uid': 1234,
        'text': 'This is a FB post!',
        'author': 'poster',
        'created_at': 9999999.0,
        'reply_to':   [1233],
        'platform':   'Facebook',
        'tags':       ['test_tag'],
        'lang':       'en'
    }


@pytest.fixture
def mock_raw_post():
    return {
        "created_time": "2012-08-19T03:00:00+0000",
        "message": "ZooWee!",
        "description": "WATCH: Video",
        "first_party": False,
        "id": "12345678_87654321",
        "link": "https://www.facebook.com/:page:/videos/12345678_87654321/",
        "name": ":page:",
        "picture": "https://scontent.xx.fbcdn.net/v/",
        "source": "https://video.xx.fbcdn.net/v/",
        "story": ":page: shared :page:'s video.",
        "type": "video",
        "updated_time": "2012-08-19T03:00:01+0000"
    }


@pytest.fixture
def mock_raw_comment():
    return {
        "created_time": "2016-12-06T03:45:00+0000",
        "id": "999999999999",
        "message": "MOCK KCOM",
        "userID": "123456",
        "from": {"name": "___", "id": "123"}
    }


@pytest.fixture
def mock_raw_comments(mock_raw_comment):
    return {
        'data': [mock_raw_comment] * 10
    }


@pytest.fixture
def mock_raw_replies(mock_raw_comment):
    nest_comment = dict(mock_raw_comment)
    mock_raw_comment['replies'] = [nest_comment] * 5
    return {
        'data': [mock_raw_comment] * 2
    }


def test_null_post(null_post):
    assert null_post.uid == 0
    assert null_post.uid == hash(null_post)
    assert null_post.text == ''
    assert null_post.author is None
    assert null_post.created_at is None
    assert null_post.reply_to == set()
    assert null_post.platform == 'Facebook'
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

    null_post.platform = 'Facebook'
    assert null_post.platform == 'Facebook'

    for tag in mock_json_post['tags']:
        null_post.add_tag(tag)
    assert null_post.tags == set(mock_json_post['tags'])

    for tag in mock_json_post['tags']:
        null_post.remove_tag(tag)
    assert null_post.tags == set()

    null_post.lang = mock_json_post['lang']
    assert null_post.lang == mock_json_post['lang']


def test_post_from_to_json(mock_json_post):
    t = FBPost.from_json(mock_json_post)

    # revert internal change of JSON
    mock_json_post['created_at'] = mock_json_post['created_at'].timestamp()

    out = t.to_json()

    for k, v in mock_json_post.items():
        assert v == out[k]

    for k, v in out.items():
        assert v == mock_json_post[k]


def test_to_from_json_null_datetime(mock_json_post):
    mock_json_post['created_at'] = None
    post = FBPost.from_json(mock_json_post)
    out = post.to_json()

    for k, v in mock_json_post.items():
        assert v == out[k]

    for k, v in out.items():
        assert v == mock_json_post[k]


def test_post_datetime_parsing():
    from datetime import datetime

    dt = '2020-12-31T23:59:59+0000'
    assert datetime(2020, 12, 31, 23, 59, 59) == FBPost.parse_datestr(dt)


def test_read_raw_post(mock_raw_post):
    from datetime import datetime

    post = FBPost.parse_raw(mock_raw_post, post_type='post')

    assert post.uid == "12345678_87654321"
    assert post.text == ":page: WATCH: Video :page: shared :page:'s video. ZooWee!"
    assert post.author is None
    assert post.created_at == datetime(2012, 8, 19, 3, 0, 0)
    assert post.reply_to == set()
    assert post.platform == 'Facebook'
    assert post.tags == {'type=video'}
    assert post.lang is None

    mock_raw_post['error'] = 'error'
    with pytest.raises(KeyError):
        FBPost.parse_raw(mock_raw_post, post_type='post')

    with pytest.raises(ValueError):
        FBPost.parse_raw(mock_raw_post, post_type='error')


def test_read_raw_comments(mock_raw_comments):
    comments = FBPost.parse_raw({}, post_type='comments', in_reply_to='12345678_87654321')
    assert len(comments) == 0

    comments = FBPost.parse_raw(mock_raw_comments, post_type='comments', in_reply_to='12345678_87654321')
    assert len(comments) == 10
    comment = comments[0]
    assert comment.uid == '999999999999'
    assert comment.text == 'MOCK KCOM'
    assert comment.author == '123456'
    assert comment.reply_to == {'12345678_87654321'}

    comments = FBPost.parse_raw(mock_raw_comments['data'], post_type='comments', in_reply_to='12345678_87654321')
    assert len(comments) == 10

    comments = FBPost.parse_raw(mock_raw_comments, post_type='comments')
    assert len(comments) == 10
    comment = comments[0]
    assert comment.uid == '999999999999'
    assert comment.text == 'MOCK KCOM'
    assert comment.author == '123456'
    assert comment.reply_to == set()

    mock_raw_comments['data'][0]['error'] = 'error'
    with pytest.raises(KeyError):
        FBPost.parse_raw(mock_raw_comments, post_type='comments', in_reply_to='12345678_87654321')


def test_read_raw_replies(mock_raw_replies):
    comments = FBPost.parse_raw({}, post_type='replies')
    assert len(comments) == 0

    comments = FBPost.parse_raw(mock_raw_replies, post_type='replies', in_reply_to='12345678_87654321')
    assert len(comments) == 12
    comment = comments[-1]
    assert comment.uid == '999999999999'
    assert comment.text == 'MOCK KCOM'
    assert comment.author == '123456'
    assert comment.reply_to == {'12345678_87654321'}

    comments = FBPost.parse_raw(mock_raw_replies['data'], post_type='replies', in_reply_to='12345678_87654321')
    assert len(comments) == 12

    mock_raw_replies['data'][0]['error'] = 'error'
    with pytest.raises(KeyError):
        FBPost.parse_raw(mock_raw_replies, post_type='replies')
