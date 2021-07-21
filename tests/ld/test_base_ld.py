from pyconversations.ld import BaseLangDetect


def test_base_lang_detection():
    det = BaseLangDetect()
    pred, conf = det.get('test text')

    assert type(pred) == str
    assert type(conf) == float
    assert pred == 'und'
    assert conf == 0.0
