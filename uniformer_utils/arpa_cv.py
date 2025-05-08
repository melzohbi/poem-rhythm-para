import random
DIPHTHONGS = {
    "AY0",
    "AY1",
    "AY2",
    "AW0",
    "AW1",
    "AW2",
    "OY0",
    "OY1",
    "OY2",
    "OW1",
    "OW2"
}

SHORT_VOWELS = {
    "AA0",
    "AE0",
    "AH0",
    "AH1",
    "AH2",
    "AO0",
    "EH0",
    "EH1",
    "EH2",
    "IH0",
    "IH1",
    "IH2",
    "UH0",
    "UW0",
    "OW0",
    "IY0",
}

LONG_VOWELS = {
    "AA1",
    "AA2",
    "AE1",
    "AE2",
    "AO1",
    "AO2",
    "EY0",
    "EY1",
    "EY2",
    "UH1",
    "UH2",
    "UW1",
    "UW2",
    "IY1",
    "IY2",
}

VOWEL_CONSONANT = {
    "ER0",
    "ER1",
    "ER2",
}

CONSONANTS = {
    "B",
    "CH",
    "D",
    "DH",
    "DX",
    "EL",
    "EM",
    "EN",
    "F",
    "G",
    "HH",
    "H",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NX",
    "NG",
    "P",
    "Q",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "V",
    "W",
    "WH",
    "Y",
    "Z",
    "ZH"
}

SPACE = {"spn"}


def classify_arpabet_phoneme(phoneme: str) -> str:
    if phoneme in SHORT_VOWELS:
        return "V"
    elif phoneme in LONG_VOWELS:
        return "VC"
    elif phoneme in DIPHTHONGS:
        return "VC"
    elif phoneme in VOWEL_CONSONANT:
        return "VC"
    elif phoneme in CONSONANTS:
        return "C"
    elif phoneme in SPACE:
        return ""
    else:
        return "?"


def convert_cv_pattern(s):
    output = []
    i = 0
    while i < len(s):
        if i == 0 and s[i] == 'V':
            output.append('1')
        elif i < len(s) - 1 and s[i] == 'C' and s[i+1] == 'V':
            output.append('1')
            i += 1  # Skip the next character
        else:
            output.append('0')
        i += 1
    return ''.join(output)


def extract_stress(text):
    text = " ".join(text)
    extracted_stress_pattern = [i for i in text if i in ['0', '1', '2']]
    # convert 0 to - and 1 to + and 2 randomly to + or -
    for i in range(len(extracted_stress_pattern)):
        if extracted_stress_pattern[i] == '0':
            extracted_stress_pattern[i] = '-'
        elif extracted_stress_pattern[i] == '1':
            extracted_stress_pattern[i] = '+'
        else:
            extracted_stress_pattern[i] = random.choice(['+', '-'])

    return ''.join(extracted_stress_pattern)
