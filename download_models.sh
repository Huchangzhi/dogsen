#!/bin/bash
# 模型下载脚本，用于在下载失败时设置为使用随机模型

echo "开始下载模型..."

# 下载红方模型
if [ -n "$RED_MODEL_URL" ]; then
    echo "尝试下载红方模型: $RED_MODEL_URL"
    if wget -O red_input_model.pth "$RED_MODEL_URL"; then
        echo "RED_MODEL_INPUT_PATH=red_input_model.pth" >> $GITHUB_ENV
        echo "红方模型下载成功"
    else
        echo "红方模型下载失败，将使用随机模型"
        # 不设置 RED_MODEL_INPUT_PATH 环境变量，让Python脚本知道不加载模型
    fi
else
    echo "未提供红方模型URL，将使用随机模型"
fi

# 下载黑方模型
if [ -n "$BLACK_MODEL_URL" ]; then
    echo "尝试下载黑方模型: $BLACK_MODEL_URL"
    if wget -O black_input_model.pth "$BLACK_MODEL_URL"; then
        echo "BLACK_MODEL_INPUT_PATH=black_input_model.pth" >> $GITHUB_ENV
        echo "黑方模型下载成功"
    else
        echo "黑方模型下载失败，将使用随机模型"
        # 不设置 BLACK_MODEL_INPUT_PATH 环境变量，让Python脚本知道不加载模型
    fi
else
    echo "未提供黑方模型URL，将使用随机模型"
fi