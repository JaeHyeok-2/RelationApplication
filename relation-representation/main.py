from korre import KorRE
from cls_model import *
from relation_extraction_model import predict
from chk_sentiment import check_sentiment_absence


korre = KorRE()
relations_emo = {}
relation = {}


def main():
    sent = "사빈이 찬영에게 클래스 확인용 예시 문장을 보여준다."
    print(korre.pororo_ner(sent))
    print(korre.ner(sent))
    print(korre.ner_sub_obj(sent))
    print(korre.ner_tagged(sent))
    print(korre.get_all_entity_pairs(sent))
    # sentence = "사빈이는 연인, 영심이한테 사랑한다고 말했다. 그러자 영심이는 감동을 받았는지 사빈이에게 눈물을 보였습니다. 사빈이는 영심이를 더 좋아하게 됐습니다. 찬영이는 그런 사빈이를 위로해줬습니다. 사빈이는 감사한 마음을 담아 찬영이에게 선물을 줬습니다."

    # for sent in sentence.split('.')[:-1]:
    #     print(check_sentiment_absence(sent))
    #     if check_sentiment_absence(sent)[0] < 1 and check_sentiment_absence(sent)[1] > -1:
    #         continue
    #     ner = korre.ner_sub_obj(sent)
    #     sub_obj = result(ner, emo_predict(sent))

    #     if f'{sub_obj[0]}-{sub_obj[1]}' not in relations_emo.keys():
    #         relations_emo[f'{sub_obj[0]}-{sub_obj[1]}'] = sub_obj[2]
    #     else:
    #         relations_emo[f'{sub_obj[0]}-{sub_obj[1]}'] += "/" + sub_obj[2]
        
    #     tagged_sent = korre.ner_tagged(sent)
    #     relation[(korre.ner_sub_obj(sent)[0][0], korre.ner_sub_obj(sent)[1][0])] = predict(tagged_sent)


    # print(relations_emo)
    # print(relation)


if __name__ == '__main__':
    main()
    