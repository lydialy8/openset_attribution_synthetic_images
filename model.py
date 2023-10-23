import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4

import cswin_transformer

swin_transformer_url = {
    "swin_tiny_224_w7_1k":"https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
    "swin_small_224_w7_1k":"https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth",
    "swin_base_224_w7_1k":"https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth",
    "swin_base_384_w12_1k":"https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth",
    "swin_tiny_224_w7_22k":"https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth",
    "swin_tiny_224_w7_22kto1k":"https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22kto1k_finetune.pth",
    "swin_small_224_w7_22k":"https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22k.pth",
    "swin_small_224_w7_22kto1k":"https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22kto1k_finetune.pth",
    "swin_base_224_w7_22k":"https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth",
    "swin_base_224_w7_22kto1k":"https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth",
    "swin_base_384_w12_22k":"https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth",
    "swin_base_384_w12_22kto1k":"https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth",
    "swin_large_224_w7_22k":"https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth",
    "swin_large_224_w7_22kto1k":"https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth",
    "swin_large_384_w12_22k":"https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth",
    "swin_large_384_w12_22kto1k":"https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth",
    
    "swin_v2_tiny_256_w8_1k":"https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window8_256.pth",
    "swin_v2_small_256_w8_1k":"https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_small_patch4_window8_256.pth",
    "swin_v2_base_256_w8_1k":"https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window8_256.pth",
    "swin_v2_tiny_256_w16_1k":"https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window16_256.pth",
    "swin_v2_small_256_w16_1k":"https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_small_patch4_window16_256.pth",
    "swin_v2_base_256_w16_1k":"https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window16_256.pth",
    "swin_v2_base_192_w12_22k":"https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window12_192_22k.pth",
    "swin_v2_base_256_w16_22kto1k":"https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth",
    "swin_v2_base_384_w24_22kto1k":"https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth",
    "swin_v2_large_192_w12_22k":"https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12_192_22k.pth",
    "swin_v2_large_256_w16_22kto1k":"https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12to16_192to256_22kto1k_ft.pth",
    "swin_v2_large_384_w24_22kto1k":"https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.pth",
}

def distance(vects):
    x, y = vects
    abs_diff = tf.math.abs(x - y)
    return abs_diff


def concatenate(vects):
    return tf.concat(vects,1)


def embedding_model(img_res, emb_size, network):
    if network=='swin':
        base_cnn = cswin_transformer.swin_transformer_v2_tiny_256_w16_1k(input_shape = (img_res, img_res, 3), include_top = False, weights = 'imagenet', conv=False)
        flatten = tf.keras.layers.Flatten()(base_cnn.output)
        dense = tf.keras.layers.Dense(emb_size, activation=None)(flatten)
        embed_net = tf.keras.models.Model(base_cnn.input, dense, name="Embedding")
        input = tf.keras.layers.Input(shape=(img_res, img_res, 3))
        embed_input = embed_net(input)
        embedding_model = tf.keras.models.Model(inputs=input, outputs=embed_input)
    else:
        base_cnn = EfficientNetB4(weights="imagenet", input_shape=(img_res, img_res, 3), include_top=False)
        flatten = tf.keras.layers.Flatten()(base_cnn.output)
        dense = tf.keras.layers.Dense(emb_size, activation=None)(flatten)
        embed_net = tf.keras.models.Model(base_cnn.input, dense, name="Embedding")
        input = tf.keras.layers.Input(shape=(img_res, img_res, 3))
        embed_input = embed_net(input)
        embedding_model = tf.keras.models.Model(inputs=input, outputs=embed_input)
    return embedding_model

def model_2denseLayers(img_res, emb_size, embedding_path, network):
    input = tf.keras.layers.Input(shape=(img_res, img_res, 3))
    ref = tf.keras.layers.Input(shape=(img_res, img_res, 3))
    embed_net = embedding_model(img_res, emb_size, network)
    embed_net.load_weights(embedding_path)
    embed_ref = embed_net(ref)
    embed_input = embed_net(input)
    embed_ref_norm = tf.keras.layers.LayerNormalization()(embed_ref)
    embed_input_norm = tf.keras.layers.LayerNormalization()(embed_input)
    merge_layer = tf.keras.layers.Lambda(distance)([embed_ref_norm, embed_input_norm])
    dn_layer_1 = tf.keras.layers.Dense(256,activation=None)(merge_layer)
    bn_layer_1 = tf.keras.layers.BatchNormalization()(dn_layer_1)
    act_layer_1 = tf.keras.layers.Activation('relu')(bn_layer_1)
    dn_layer_2 = tf.keras.layers.Dense(64,activation=None)(act_layer_1)
    bn_layer_2 = tf.keras.layers.BatchNormalization()(dn_layer_2)
    act_layer_2 = tf.keras.layers.Activation('relu')(bn_layer_2)
    logits_layer = tf.keras.layers.Dense(1,activation=None,name='logits')(act_layer_2)
    model = tf.keras.models.Model(inputs=[input, ref], outputs=logits_layer)
    return model