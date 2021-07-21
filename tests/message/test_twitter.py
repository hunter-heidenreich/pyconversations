import pytest

from pyconversations.message import Tweet


@pytest.fixture
def null_tweet():
    """Returns a Tweet with a UID of 0"""
    return Tweet(uid=0)


@pytest.fixture
def mock_json_tweet():
    """Returns a mock json of a cached tweet"""
    return {
        'uid': 1234,
        'text': 'This is a tweet! @Twitter',
        'author': 'tweeter1',
        'created_at': 9999999.0,
        'reply_to':   [1233],
        'platform':   'Twitter',
        'tags':       ['test_tag'],
        'lang':       'en'
    }


@pytest.fixture
def mock_tweet(mock_json_tweet):
    """Returns a mock tweet object"""
    return Tweet(**mock_json_tweet)


@pytest.fixture
def mock_raw_tweet():
    return {
        "created_at": "Wed Jun 17 21:49:58 +0000 2020",
        "id": 123456789101112131415,
        "id_str": "123456789101112131415",
        "full_text": "@USER1 @USER2 This is ridiculous https://t.co/ https://url.co/",
        "truncated": False,
        "display_text_range": [26, 88],
        "entities": {
            "hashtags": [],
            "symbols": [],
            "user_mentions": [
                {
                    "screen_name": "USER1",
                    "name": "USER1",
                    "id": 12345,
                    "id_str": "12345",
                    "indices": [0, 8]
                }, {
                    "screen_name": "USER2",
                    "name": "USER2",
                    "id": 54321,
                    "id_str": "54321",
                    "indices": [9, 25]
                }
            ],
            "urls": [{
                "url": "https://url.co/",
                "expanded_url": "https://www.youtube.com/channel/",
                "display_url": "youtube.com/channel/\u2026",
                "indices": [0, 23]
            }],
            "media": [{
                "id": 999999999999,
                "id_str": "999999999999",
                "indices": [73, 96],
                "media_url": "http://pbs.twimg.com/tweet_video_thumb/.jpg",
                "media_url_https": "https://pbs.twimg.com/tweet_video_thumb/.jpg",
                "url": "https://t.co/",
                "display_url": "pic.twitter.com/",
                "expanded_url": "https://twitter.com//status//photo/1",
                "type": "photo",
                "sizes": {
                    "thumb":  {"w": 150, "h": 150, "resize": "crop"},
                    "large":  {"w": 498, "h": 336, "resize": "fit"},
                    "medium": {"w": 498, "h": 336, "resize": "fit"},
                    "small":  {"w": 498, "h": 336, "resize": "fit"}
                }
            }]
        },
        "metadata": {
            "iso_language_code": "en",
            "result_type": "recent"
        },
        "source": "<a href=\"https://mobile.twitter.com\" rel=\"nofollow\">Twitter Web App</a>",
        "in_reply_to_status_id": 123456789101112131414,
        "in_reply_to_status_id_str": "123456789101112131414",
        "in_reply_to_user_id": 12345,
        "in_reply_to_user_id_str": "12345",
        "in_reply_to_screen_name": "USER1",
        "user": {
            "id": 999999,
            "id_str": "999999",
            "name": "USER3",
            "screen_name": "USER3",
            "location": "",
            "description": "",
            "url": None,
            "entities": {
                "description": {"urls": []}
            },
            "protected": False,
            "followers_count": 100,
            "friends_count": 100,
            "listed_count": 0,
            "created_at": "Wed Feb 11 20:35:22 +0000 2018",
            "favourites_count": 500,
            "utc_offset": None,
            "time_zone": None,
            "geo_enabled": False,
            "verified": False,
            "statuses_count": 135,
            "lang": None,
            "contributors_enabled": False,
            "is_translator": False,
            "is_translation_enabled": False,
            "profile_background_color": "C0DEED",
            "profile_background_image_url": "http://abs.twimg.com/images/themes/theme1/bg.png",
            "profile_background_image_url_https": "https://abs.twimg.com/images/themes/theme1/bg.png",
            "profile_background_tile": False,
            "profile_image_url": "http://pbs.twimg.com/profile_images/123/normal.jpeg",
            "profile_image_url_https": "https://pbs.twimg.com/profile_images/123/normal.jpeg",
            "profile_link_color": "1DA1F2",
            "profile_sidebar_border_color": "C0DEED",
            "profile_sidebar_fill_color": "DDEEF6",
            "profile_text_color": "333333",
            "profile_use_background_image": True,
            "has_extended_profile": False,
            "default_profile": True,
            "default_profile_image": False,
            "following": None,
            "follow_request_sent": None,
            "notifications": None,
            "translator_type": "none"
        },
        "geo": None,
        "coordinates": None,
        "place": None,
        "contributors": None,
        "is_quote_status": False,
        "retweet_count": 0,
        "favorite_count": 0,
        "favorited": False,
        "retweeted": False,
        "lang": "en"
    }


def test_null_tweet(null_tweet):
    assert null_tweet.uid == 0
    assert null_tweet.uid == hash(null_tweet)
    assert null_tweet.text == ''
    assert null_tweet.author is None
    assert null_tweet.created_at is None
    assert null_tweet.reply_to == set()
    assert null_tweet.platform == 'Twitter'
    assert null_tweet.tags == set()
    assert null_tweet.lang is None


def test_update_tweet(null_tweet, mock_json_tweet):
    null_tweet.text = mock_json_tweet['text']
    assert null_tweet.text == mock_json_tweet['text']

    null_tweet.author = mock_json_tweet['author']
    assert null_tweet.author == mock_json_tweet['author']

    null_tweet.created_at = mock_json_tweet['created_at']
    assert null_tweet.created_at.timestamp() == mock_json_tweet['created_at']

    for uid in mock_json_tweet['reply_to']:
        null_tweet.add_reply_to(uid)
    assert null_tweet.reply_to == set(mock_json_tweet['reply_to'])

    for uid in mock_json_tweet['reply_to']:
        null_tweet.remove_reply_to(uid)
    assert null_tweet.reply_to == set()

    null_tweet.platform = 'Twitter'
    assert null_tweet.platform == 'Twitter'

    for tag in mock_json_tweet['tags']:
        null_tweet.add_tag(tag)
    assert null_tweet.tags == set(mock_json_tweet['tags'])

    for tag in mock_json_tweet['tags']:
        null_tweet.remove_tag(tag)
    assert null_tweet.tags == set()

    null_tweet.lang = mock_json_tweet['lang']
    assert null_tweet.lang == mock_json_tweet['lang']


def test_tweet_from_to_json(mock_json_tweet):
    t = Tweet.from_json(mock_json_tweet)

    # revert internal change of JSON
    mock_json_tweet['created_at'] = mock_json_tweet['created_at'].timestamp()

    out = t.to_json()

    for k, v in mock_json_tweet.items():
        assert v == out[k]

    for k, v in out.items():
        assert v == mock_json_tweet[k]


def test_to_from_json_null_datetime(mock_json_tweet):
    mock_json_tweet['created_at'] = None
    post = Tweet.from_json(mock_json_tweet)
    out = post.to_json()

    for k, v in mock_json_tweet.items():
        assert v == out[k]

    for k, v in out.items():
        assert v == mock_json_tweet[k]


def test_tweet_mentions(mock_tweet, null_tweet):
    assert mock_tweet.get_mentions() == {'Twitter', 'tweeter1'}
    assert null_tweet.get_mentions() == set()


def test_post_redaction(mock_tweet):
    mock_tweet.redact({'Twitter': 'NAME1'})
    assert mock_tweet.text == 'This is a tweet! @NAME1'

    mock_tweet.redact({'tweeter1': 'NAME2'})
    assert mock_tweet.author == 'NAME2'

    mock_tweet.text = ''
    mock_tweet.redact({'tweeter1': 'NAME2'})
    assert mock_tweet.author == 'NAME2'
    assert mock_tweet.text == ''


def test_tweet_repr(mock_tweet):
    assert mock_tweet.__repr__() == 'Tweet(Twitter::tweeter1::9999999.0::This is a tweet! @Twitter::tags=test_tag)'


def test_tweet_datetime_parsing(null_tweet):
    from datetime import datetime

    dt = 'Thu Dec 31 23:59:59 +0000 2020'
    assert datetime(2020, 12, 31, 23, 59, 59) == Tweet.parse_datestr(dt)

    null_tweet.created_at = dt
    assert datetime(2020, 12, 31, 23, 59, 59) == null_tweet.created_at

    with pytest.raises(TypeError):
        null_tweet.created_at = {}


def test_read_raw_tweet(mock_raw_tweet):
    from datetime import datetime

    ts = Tweet.parse_raw(mock_raw_tweet)
    assert len(ts) == 1

    t = ts[0]
    assert t.uid == 123456789101112131415
    assert t.text == '@USER1 @USER2 This is ridiculous pic.twitter.com/ https://www.youtube.com/channel/'
    assert t.author == 'USER3'
    assert t.created_at == datetime(2020, 6, 17, 21, 49, 58)  # Wed Jun 17 21:49:58 +0000 2020
    assert t.reply_to == {123456789101112131414}
    assert t.platform == 'Twitter'
    assert t.tags == set()
    assert t.lang == 'en'

    mock_raw_tweet['text'] = mock_raw_tweet['full_text']
    del mock_raw_tweet['full_text']
    ts = Tweet.parse_raw(mock_raw_tweet)
    assert len(ts) == 1

    with pytest.raises(KeyError):
        mock_raw_tweet['entities']['error'] = 'error'
        Tweet.parse_raw(mock_raw_tweet)

    del mock_raw_tweet['entities']
    ts = Tweet.parse_raw(mock_raw_tweet)
    assert len(ts) == 1

    mock_raw_tweet['is_quote_status'] = True
    mock_raw_tweet['quoted_status_id'] = 123456789101112131415
    mock_raw_tweet['quoted_status'] = dict(mock_raw_tweet)
    ts = Tweet.parse_raw(mock_raw_tweet)
    assert len(ts) == 2


def test_read_raw_no_text(mock_raw_tweet):
    del mock_raw_tweet['full_text']
    ts = Tweet.parse_raw(mock_raw_tweet)
    assert len(ts) == 0


def test_read_raw_error(mock_raw_tweet):
    mock_raw_tweet['fake'] = 'test'
    with pytest.raises(KeyError):
        Tweet.parse_raw(mock_raw_tweet)


def test_tweet_lang_detect(mock_raw_tweet):
    mock_raw_tweet['lang'] = None
    ts = Tweet.parse_raw(mock_raw_tweet, lang_detect=True)
    assert len(ts) == 1
    assert ts[0].lang == 'en'

    mock_raw_tweet['full_text'] = 'a09fuda-s9hviajosd9asvu[v9jxz0c-09k9-'
    mock_raw_tweet['lang'] = None
    ts = Tweet.parse_raw(mock_raw_tweet, lang_detect=True)
    assert len(ts) == 1
    assert ts[0].lang == 'und'


def test_tokenization(mock_tweet):
    assert mock_tweet.tokens == ['This', ' ', 'is', ' ', 'a', ' ', 'tweet', '!', ' ', '@Twitter']
