import re
from typing import List


def split_to_jamo(string: str) -> List[str]:
    # 유니코드 한글 시작 : 44032, 끝 : 55199
    _base_code = 44032
    _chosung = 588
    _jungsung = 28
    # 초성 리스트. 00 ~ 18
    _chosung_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ',
                     'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ',
                     'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    # 중성 리스트. 00 ~ 20
    _jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ',
                      'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ',
                      'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
    # 종성 리스트. 00 ~ 27 + 1(1개 없음)
    _jongsung_list = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ',
                      'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ',
                      'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    def split(sequence):
        split_string = list(sequence)
        list_of_tokens = []
        for char in split_string:
            # 한글 여부 check 후 분리
            if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', char) is not None:
                if ord(char) < _base_code:
                    list_of_tokens.append(char)
                    continue

                char_code = ord(char) - _base_code
                alphabet1 = int(char_code / _chosung)
                list_of_tokens.append(_chosung_list[alphabet1])
                alphabet2 = int((char_code - (_chosung * alphabet1)) / _jungsung)
                list_of_tokens.append(_jungsung_list[alphabet2])
                alphabet3 = int((char_code - (_chosung * alphabet1) - (_jungsung * alphabet2)))

                if alphabet3 != 0:
                    list_of_tokens.append(_jongsung_list[alphabet3])
            else:
                list_of_tokens.append(char)
        return list_of_tokens

    return split(string)
