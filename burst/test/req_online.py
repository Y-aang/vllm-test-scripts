import requests
import json

url = "http://localhost:8000/generate"
headers = {"Content-Type": "application/json"}
data = {
    "prompt": "你好，介绍一下你自己",
    # "prompt": "q j m v i x q z l d s l i w s d i r s g i i s j n s p y x s r v j c y r q m o c f g o g m p v b u h p r x q x r m y p y b c g m f p s k q q n i p l t y n h y h d h f y k d y g k d m y w q s n s b q v r k z s o x i e m u d l c m t u b i r l q k o v n b y g t h h y v h c h p n m k j p u o i d v p b f k x b w u b q n q s k k t f j i q p l c d w g f n c q g s h l k l x b q m m d n p l t g d i p k w m g r g g b j v g p p k j q w d n y d d w h k d v q e v u n s w b h j x v z h x z o k k r i p p r p s l v o i d s d k o i p k j s e y q g p r f x d e z o p o j u o q u j h v c w y p x y r e u t i o q j y w l p w c e g d u m l s o p o t p r l e k x p p f l e f w w u h x w o l r u b j f z h j d h r c u h r d u y g m x s u m k y x w u f x j e r q t g p k t d q n r r h p",
    "n": 1,
    "temperature": 0.0,
    "max_tokens": 16,
    "stream": True
}
print('posting request...')
response = requests.post(url, headers=headers, json=data, stream=True)
print('response:', type(response))
for line in response.iter_lines():
    if line:
        print(json.loads(line.decode("utf-8")))