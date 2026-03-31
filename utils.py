import os
import sys
import random
import logging
import yaml
import torch
import pickle

from datetime import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from matplotlib.lines import Line2D
import colorsys


logging.basicConfig(force=True, level=logging.INFO)

def set_seed(seed):
    """Sets random seed everywhere."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # use determinisitic algorithm
    print("Seed set", seed)

def get_config_args(parser, yaml_file_path, dataset):
    # Load YAML file
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)[dataset]

    # Add arguments with defaults from YAML
    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                parser.add_argument(f'--{key}_{sub_key}', type=type(sub_value), default=sub_value)
        elif isinstance(value, list):
            parser.add_argument(f'--{key}', type=str, nargs='+', default=value)
        else:
            parser.add_argument(f'--{key}', type=type(value), default=value)

    # Parse arguments
    args = parser.parse_args()

    return args


def save_pkl(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)

def make_route(dir_path, file_name=None):
    # Full path for the directory
    absolute_path = os.path.join(os.getcwd(), dir_path)

    # Check if the directory exists, create it if it doesn't
    if not os.path.exists(absolute_path):
        os.makedirs(absolute_path)

    # only for making directory
    if file_name is None:
        return

    # Full path for the file inside the directory
    file_path = os.path.join(absolute_path, file_name)

    # Check if the file already exists
    if os.path.exists(file_path):
        # Get the current date and time
        current_datetime = dt.now().strftime('%Y-%m-%d-%H-%M-%S')
        # Define the new filename with the current date and time
        title, extension = os.path.splitext(file_name)
        new_file_name = f'{title}-backup-{current_datetime}-{extension}'
        # Rename the existing file
        new_file_path = os.path.join(absolute_path, new_file_name)
        os.rename(file_path, new_file_path)

    # Create a new file (or open the file if it somehow already exists) and write something to it
    f = open(file_path, 'w')
    f.close()

    return


def plot_and_save_loss(train_losses, val_losses, test_losses, filename):
    """
    Generates a line graph comparing training and validation losses over epochs and saves the figure to a file.

    Parameters:
    - train_losses (list of float): The training losses for each epoch.
    - val_losses (list of float): The validation losses for each epoch.
    - filename (str): The name of the file to save the plot. Defaults to 'loss_comparison.png'.

    Returns:
    - None
    """
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.plot(epochs, test_losses, label = 'Test Loss')

    plt.title('Training vs Validation vs Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)
    plt.close('all')  # close instead of show — avoids blocking on headless servers
    print(f"Plot saved as {filename}")


def alignment_save_scatter(result, label, save_path, dr_type=1):
    if dr_type == 0:
        dr_name = 'UMAP'
    elif dr_type == 1:
        dr_name = 'TSNE'
    else:
        dr_name = ''


    a, t, v = result

    colors = [
        '#1f77b4',  # Tableau Blue
        '#2ca02c',  # Tableau Green
        '#d62728',  # Tableau Red
    ]

    plt.clf()

    # 🔽🔽🔽 --- 수정/추가된 부분 시작 --- 🔽🔽🔽
    # 현재 figure와 axes 객체를 가져옵니다.
    fig, ax = plt.subplots(figsize=(8, 8))  # figsize로 그림 크기 조절 가능

    # X축과 Y축의 눈금을 모두 제거합니다.
    ax.set_xticks([])
    ax.set_yticks([])

    # X축과 Y축의 레이블을 제거합니다.
    ax.set_xlabel('')
    ax.set_ylabel('')

    # 그래프의 테두리(spines)를 모두 보이지 않게 설정합니다.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # 🔼🔼🔼 --- 수정/추가된 부분 끝 --- 🔼🔼🔼

    _s = 4.
    # plt.scatter를 ax.scatter로 변경합니다.
    c1 = ax.scatter(a[:, 0], a[:, 1], marker="o", color=colors[0], s=_s, label='Audio')
    c2 = ax.scatter(t[:, 0], t[:, 1], marker="o", color=colors[1], s=_s, label='Text')
    c3 = ax.scatter(v[:, 0], v[:, 1], marker="o", color=colors[2], s=_s, label='Visual')

    ax.legend(handles=(c1, c2, c3),
              labels=("Audio", "Text", "Visual"))

    # 🔽🔽🔽 --- savefig 수정 --- 🔽🔽🔽
    # facecolor 옵션을 제거하면 기본 흰색 배경으로 저장됩니다.
    fig.savefig(
        save_path + '/alignment.png',
        dpi=300, bbox_inches='tight')

    plt.close(fig)  # figure 객체를 닫아줍니다.
    return




# --- 1. 색상 밝기를 미세하게 조절하는 헬퍼 함수 ---
def adjust_color_lightness(hex_color, factor):
    """HEX 색상을 입력받아 밝기를 조절한 뒤 다시 HEX로 반환"""
    # HEX to RGB
    rgb = tuple(int(hex_color.lstrip('#')[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    # RGB to HLS
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # Lightness 조절 (factor > 1.0: 밝게, factor < 1.0: 어둡게)
    new_l = max(0, min(1, l * factor))
    # HLS to RGB
    new_rgb = colorsys.hls_to_rgb(h, new_l, s)
    # RGB to HEX
    return '#%02x%02x%02x' % tuple(int(c * 255) for c in new_rgb)

def alignment_line_class_meld_save_scatter(result, label, save_path, dr_type=1):

    a, t, v = result

    # --- 2. 스타일 설정 ---
    num_classes = 5
    class_labels = ["Neutral", "Surprise", "Sadness", "Joy", "Anger"]
    modality_labels = ["Audio", "Text", "Visual"]

    # 클래스별 기본 색상
    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # 모달리티별 색상 밝기 조절 (Audio: 약간 밝게, Text: 그대로, Visual: 약간 어둡게)
    color_palettes = [
        (adjust_color_lightness(c, 1.), adjust_color_lightness(c, 1.4), adjust_color_lightness(c, 0.6))
        for c in base_colors
    ]

    markers = ['o', 's', '^']

    # --- 3. 그래프 기본 설정 ---
    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # --- 4. 라인 연결 및 Scatter 그리기 ---
    _s = 12.

    # 라인 연결 (zorder=1, 가장 아래)
    for i in range(len(a)):
        points_x = [a[i, 0], t[i, 0], v[i, 0], a[i, 0]]
        points_y = [a[i, 1], t[i, 1], v[i, 1], a[i, 1]]
        ax.plot(points_x, points_y, color='gray', linewidth=0.5, alpha=0.5, zorder=1)

    # Scatter 그리기 (zorder=2, 라인 위)
    for i in range(num_classes):
        indices = np.where(label == i)[0]
        if len(indices) == 0:
            continue

        current_palette = color_palettes[i]

        # Audio (약간 밝게, 동그라미)
        ax.scatter(a[indices, 0], a[indices, 1], marker=markers[0], color=current_palette[0], s=_s, zorder=2)
        # Text (기본색, 네모)
        ax.scatter(t[indices, 0], t[indices, 1], marker=markers[1], color=current_palette[1], s=_s, zorder=2)
        # Visual (약간 어둡게, 세모)
        ax.scatter(v[indices, 0], v[indices, 1], marker=markers[2], color=current_palette[2], s=_s, zorder=2)

    # --- 5. 커스텀 범례 생성 ---
    # 클래스(색상) 범례
    color_handles = [Line2D([0], [0], marker='o', color='w', label=class_labels[i],
                            markerfacecolor=base_colors[i], markersize=8) for i in range(num_classes)]
    legend1 = ax.legend(handles=color_handles, title="Emotions", loc='upper right', fontsize=15, title_fontsize=15)
    ax.add_artist(legend1)

    # 모달리티(마커) 범례
    marker_handles = [Line2D([0], [0], marker=m, color='gray', label=modality_labels[i],
                             linestyle='None', markersize=8) for i, m in enumerate(markers)]
    ax.legend(handles=marker_handles, title="Modalities", loc='lower right', fontsize=15, title_fontsize=15)

    # --- 6. 저장 ---
    fig.savefig(
        save_path + '/alignment_by_class_intensity_lines.png',
        dpi=300, bbox_inches='tight')

    plt.close(fig)
    return


def alignment_line_class_save_scatter(result, label, save_path, dr_type=1):

    a, t, v = result

    # --- 2. 스타일 설정 ---
    num_classes = 6
    class_labels = ["Happy", "Sad", "Neutral", "Angry", "Excited", "Frustrated"]
    modality_labels = ["Audio", "Text", "Visual"]

    # 클래스별 기본 색상
    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # 모달리티별 색상 밝기 조절 (Audio: 약간 밝게, Text: 그대로, Visual: 약간 어둡게)
    color_palettes = [
        (adjust_color_lightness(c, 1.), adjust_color_lightness(c, 1.4), adjust_color_lightness(c, 0.6))
        for c in base_colors
    ]

    markers = ['o', 's', '^']

    # --- 3. 그래프 기본 설정 ---
    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # --- 4. 라인 연결 및 Scatter 그리기 ---
    _s = 12.

    # 라인 연결 (zorder=1, 가장 아래)
    for i in range(len(a)):
        points_x = [a[i, 0], t[i, 0], v[i, 0], a[i, 0]]
        points_y = [a[i, 1], t[i, 1], v[i, 1], a[i, 1]]
        ax.plot(points_x, points_y, color='gray', linewidth=0.5, alpha=0.5, zorder=1)

    # Scatter 그리기 (zorder=2, 라인 위)
    for i in range(num_classes):
        indices = np.where(label == i)[0]
        if len(indices) == 0:
            continue

        current_palette = color_palettes[i]

        # Audio (약간 밝게, 동그라미)
        ax.scatter(a[indices, 0], a[indices, 1], marker=markers[0], color=current_palette[0], s=_s, zorder=2)
        # Text (기본색, 네모)
        ax.scatter(t[indices, 0], t[indices, 1], marker=markers[1], color=current_palette[1], s=_s, zorder=2)
        # Visual (약간 어둡게, 세모)
        ax.scatter(v[indices, 0], v[indices, 1], marker=markers[2], color=current_palette[2], s=_s, zorder=2)

    # --- 5. 커스텀 범례 생성 ---
    # 클래스(색상) 범례
    color_handles = [Line2D([0], [0], marker='o', color='w', label=class_labels[i],
                            markerfacecolor=base_colors[i], markersize=8) for i in range(num_classes)]
    legend1 = ax.legend(handles=color_handles, title="Emotions", loc='upper right', fontsize=16, title_fontsize=16)
    ax.add_artist(legend1)

    # 모달리티(마커) 범례
    marker_handles = [Line2D([0], [0], marker=m, color='gray', label=modality_labels[i],
                             linestyle='None', markersize=8) for i, m in enumerate(markers)]
    ax.legend(handles=marker_handles, title="Modalities", loc='lower right', fontsize=16, title_fontsize=16)

    # --- 6. 저장 ---
    fig.savefig(
        save_path + '/alignment_by_class_intensity_lines.png',
        dpi=300, bbox_inches='tight')

    plt.close(fig)
    return

def alignment_line_save_scatter(result, label, save_path, dr_type=1):
    if dr_type == 0:
        dr_name = 'UMAP'
    elif dr_type == 1:
        dr_name = 'TSNE'
    else:
        dr_name = ''

    a, t, v = result

    colors = [
        '#1f77b4',  # Tableau Blue
        '#2ca02c',  # Tableau Green
        '#d62728',  # Tableau Red
    ]

    plt.clf()

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # 🔽🔽🔽 --- 라인 연결 코드 추가 시작 --- 🔽🔽🔽
    # 각 샘플 인덱스(i)에 대해 반복
    for i in range(len(a)):
        # a[i], t[i], v[i] 세 점의 x, y 좌표
        points_x = [a[i, 0], t[i, 0], v[i, 0], a[i, 0]]
        points_y = [a[i, 1], t[i, 1], v[i, 1], a[i, 1]]

        # 세 점을 잇는 선 그리기
        # 선이 점을 가리지 않도록 점보다 먼저 그립니다.
        ax.plot(points_x, points_y, color='gray', linewidth=0.1, alpha=0.5)
    # 🔼🔼🔼 --- 라인 연결 코드 추가 끝 --- 🔼🔼🔼

    _s = 4.
    c1 = ax.scatter(a[:, 0], a[:, 1], marker="o", color=colors[0], s=_s, label='Audio', zorder=2)
    c2 = ax.scatter(t[:, 0], t[:, 1], marker="o", color=colors[1], s=_s, label='Text', zorder=2)
    c3 = ax.scatter(v[:, 0], v[:, 1], marker="o", color=colors[2], s=_s, label='Visual', zorder=2)

    ax.legend(handles=(c1, c2, c3),
              labels=("Audio", "Text", "Visual"))

    fig.savefig(
        save_path + '/alignment.png',
        dpi=300, bbox_inches='tight')

    plt.close(fig)
    return


def discrimination_save_scatter(result, label, save_path, dr_type=1, dataset_name="iemocap", prefix="discrimination"):
    if dr_type == 0:
        dr_name = 'UMAP'
    elif dr_type == 1:
        dr_name = 'TSNE'
    else:
        dr_name = ''

    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    _s = 12.

    if dataset_name == "iemocap":
        classes = [0, 1, 2, 3, 4, 5]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        labels = ["Happy", "Sad", "Neutral", "Angry", "Excited", "Frustrated"]
    elif dataset_name == "iemocap_4":
        classes = [0, 1, 2, 3]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        labels = ["Happy", "Sad", "Neutral", "Angry"]
    elif dataset_name == "meld":
        classes = [0, 1, 3, 4, 6]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        labels = ["Neutral", "Surprise", "Sadness", "Joy", "Anger"]
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    handles = []
    for c_idx, color, label_str in zip(classes, colors, labels):
        indices = np.where(label == c_idx)[0]
        if len(indices) > 0:
            scatter = ax.scatter(result[indices, 0], result[indices, 1], marker="o", color=color, s=_s, label=label_str)
            handles.append(scatter)

    if handles:
        ax.legend(handles=handles, labels=[h.get_label() for h in handles], title="Emotions", loc='best', fontsize=12, title_fontsize=14)
 
    save_filename = os.path.join(save_path, f"{prefix}_{dr_name.lower()}.png")
    fig.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Class Discrimination plot saved as {save_filename}")
    return

def plot_and_save_confusion_matrix(golds, preds, class_labels:dict, save_path="confusion_matrix.png",
                                   fontsize=12):
    """
    Normalized Confusion Matrix를 생성하고 타이트한 형태로 저장하는 함수 (폰트 조정 가능)

    Args:
        golds (list or np.array): 실제 레이블 (정답)
        preds (list or np.array): 예측 레이블
        class_labels (list): 클래스 이름 리스트
        save_path (str): 저장할 이미지 파일 경로
        fontsize (int): 폰트 크기
    """
    # ✅ Confusion Matrix 계산 (normalize='true'를 사용하여 행 단위 정규화)
    cm = confusion_matrix(golds, preds, normalize='true')

    # ✅ Seaborn 스타일 설정
    sns.set(font_scale=1.2)  # 전체 폰트 크기 스케일링

    # ✅ 시각화 (Seaborn을 이용한 Confusion Matrix Plot)
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_labels.keys(), yticklabels=class_labels.keys(),
                     annot_kws={"size": fontsize})  # ✅ 셀 내부 숫자 폰트 크기 조정

    # ✅ 축 라벨 및 제목 폰트 크기 설정
    plt.xlabel("Predicted Labels", fontsize=fontsize)
    plt.ylabel("True Labels", fontsize=fontsize)
    plt.title("Normalized Confusion Matrix", fontsize=fontsize)

    # ✅ X/Y 눈금 폰트 크기 조정
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)

    # ✅ Confusion Matrix 타이트하게 저장 (bbox_inches="tight" 추가)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()  # 화면에 출력
    plt.close()  # 메모리 절약을 위해 close()
    print(f"✅ Confusion Matrix saved to {save_path} (tight box format)")



def plot_modality_alignment(result, label, save_dir, dr_name, dataset_name="iemocap", prefix="modality_alignment"):
    """
    Plots modality alignment space with distinct colors for modalities.
    No class discrimination is shown (all points are circles).
    """
    a, t, v = result
    
    modality_labels = ["Audio", "Text", "Visual"]
    modality_colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green
    
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    _s = 20. # Marker size
    
    # Audio
    ax.scatter(a[:, 0], a[:, 1], marker='o', color=modality_colors[0], s=_s, alpha=0.8, zorder=2)
    # Text
    ax.scatter(t[:, 0], t[:, 1], marker='o', color=modality_colors[1], s=_s, alpha=0.8, zorder=2)
    # Visual
    ax.scatter(v[:, 0], v[:, 1], marker='o', color=modality_colors[2], s=_s, alpha=0.8, zorder=2)
        
    # Legend 1: Modalities (Colors)
    color_handles = [Line2D([0], [0], marker='o', color='w', label=modality_labels[i],
                            markerfacecolor=modality_colors[i], markersize=10) for i in range(len(modality_labels))]
    legend1 = ax.legend(handles=color_handles, title="Modalities", loc='best', fontsize=24, title_fontsize=24)
    ax.add_artist(legend1)
    
    plt.title(f"{dataset_name.upper()} Modality Alignment ({dr_name})", fontsize=16, pad=20)
    
    os.makedirs(save_dir, exist_ok=True)
    save_filename = os.path.join(save_dir, f"{prefix}_{dr_name.lower().replace('-', '')}.png")
    fig.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Modality Alignment plot saved as {save_filename}")

def plot_modality_alignment_dual(result, label, save_dir, dr_name, dataset_name="iemocap", prefix="modality_alignment_dual"):
    """
    Dual-encoded modality alignment plot.
    Colors = Emotion classes (for class discrimination visibility).
    Shapes = Modalities (circle=Audio, square=Text, triangle=Visual).
    """
    a, t, v = result

    if dataset_name == "iemocap":
        classes = [0, 1, 2, 3, 4, 5]
        class_colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4']
        class_labels = ["Happy", "Sad", "Neutral", "Angry", "Excited", "Frustrated"]
    elif dataset_name == "iemocap_4":
        classes = [0, 1, 2, 3]
        class_colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231']
        class_labels = ["Happy", "Sad", "Neutral", "Angry"]
    else:  # meld
        classes = [0, 1, 3, 4, 6]
        class_colors = ['#4363d8', '#f58231', '#3cb44b', '#e6194b', '#911eb4']
        class_labels = ["Neutral", "Surprise", "Sadness", "Joy", "Anger"]

    modality_labels = ["Audio", "Text", "Visual"]
    modality_markers = ['o', 's', '^']       # circle, square, triangle
    modality_sizes   = [22., 20., 28.]       # slightly tweaked for visual balance
    modality_data    = [a, t, v]

    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Scatter per (class, modality)
    for c_idx, color, c_label in zip(classes, class_colors, class_labels):
        indices = np.where(label == c_idx)[0]
        if len(indices) == 0:
            continue
        for m_idx, (data, marker, ms) in enumerate(zip(modality_data, modality_markers, modality_sizes)):
            ax.scatter(
                data[indices, 0], data[indices, 1],
                marker=marker, color=color, s=ms, alpha=0.8,
                edgecolors='none', zorder=2,
            )

    # Legend 1: Emotions (Colors)
    emotion_handles = [
        Line2D([0], [0], marker='o', color='w', label=c_label,
               markerfacecolor=color, markersize=10, linestyle='None')
        for color, c_label in zip(class_colors, class_labels)
    ]
    legend1 = ax.legend(handles=emotion_handles, title="Emotions",
                        loc='upper left', fontsize=14, title_fontsize=16)
    ax.add_artist(legend1)

    # Legend 2: Modalities (Shapes)
    marker_handles = [
        Line2D([0], [0], marker=marker, color='w', label=modality_labels[i],
               markerfacecolor='gray', markersize=12, linestyle='None')
        for i, marker in enumerate(modality_markers)
    ]
    ax.legend(handles=marker_handles, title="Modalities",
             loc='lower right', fontsize=14, title_fontsize=16)

    plt.title(f"{dataset_name.upper()} Modality Alignment — Dual Encoded ({dr_name})",
              fontsize=16, pad=20)

    os.makedirs(save_dir, exist_ok=True)
    save_filename = os.path.join(
        save_dir, f"{prefix}_{dr_name.lower().replace('-', '')}.png")
    fig.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Modality Alignment (Dual) plot saved as {save_filename}")

def plot_independent_modality_distributions(results, label, save_dir, dr_name, dataset_name="iemocap", prefix="independent_distribution"):
    """
    Plots independent modal reductions in a 1x3 grid side-by-side.
    Colors = Emotions, Shapes = Modalities.
    """
    a, t, v = results
    
    if dataset_name == "iemocap":
        classes = [0, 1, 2, 3, 4, 5]
        class_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        class_labels = ["Happy", "Sad", "Neutral", "Angry", "Excited", "Frustrated"]
    elif dataset_name == "iemocap_4":
        classes = [0, 1, 2, 3]
        class_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        class_labels = ["Happy", "Sad", "Neutral", "Angry"]
    else: # meld
        classes = [0, 1, 3, 4, 6]
        class_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        class_labels = ["Neutral", "Surprise", "Sadness", "Joy", "Anger"]
        
    modality_labels = ["Audio", "Text", "Visual"]
    modality_markers = ['o', 's', '^'] # Circle, Square, Triangle
    
    plt.clf()
    fig, axes = plt.subplots(1, 3, figsize=(24, 8)) # 1x3 Grid format
    
    _s = 25.
    handles_emotions = []
    
    for i, (ax, data, marker, mod_name) in enumerate(zip(axes, [a, t, v], modality_markers, modality_labels)):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        for c_idx, color, label_str in zip(classes, class_colors, class_labels):
            indices = np.where(label == c_idx)[0]
            if len(indices) == 0: continue
            scatter = ax.scatter(data[indices, 0], data[indices, 1], marker=marker, color=color, s=_s, alpha=0.8, label=label_str)
            if i == 0: # Populate legend only from one axis
                handles_emotions.append(scatter)
                
    # Generate Legend for Modalities Shapes
    marker_handles = [Line2D([0], [0], marker=m, color='w', label=modality_labels[idx],
                             markerfacecolor='gray', markersize=12) for idx, m in enumerate(modality_markers)]
                             
    # Mount legends strictly on the right external frame to not overlay data clusters or alter grid widths improperly
    legend1 = axes[2].legend(handles=handles_emotions, title="Emotions", loc='upper left', bbox_to_anchor=(1, 1), fontsize=14, title_fontsize=16)
    axes[2].add_artist(legend1)
    
    axes[2].legend(handles=marker_handles, title="Modalities", loc='lower left', bbox_to_anchor=(1, 0), fontsize=14, title_fontsize=16)
    
    os.makedirs(save_dir, exist_ok=True)
    save_filename = os.path.join(save_dir, f"{prefix}_{dr_name.lower().replace('-', '')}.png")
    fig.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Independent Modality Distribution plot saved as {save_filename}")

def get_logger(filepath: str, level=logging.INFO):
    logger = logging.getLogger(__name__)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    fileHandler = logging.FileHandler(filepath)
    streamHandler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        fmt='[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s'
    )
    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    return logger

def linear_cka(X, Y):
    """
    Computes Linear Centered Kernel Alignment (CKA).
    Measures Representational Geometry Isomorphism (1.0 = highly correlated structures).
    """
    # Center columns
    X_c = X - np.mean(X, axis=0)
    Y_c = Y - np.mean(Y, axis=0)
    
    # Calculate similarity matrices
    C_XX = X_c @ X_c.T
    C_YY = Y_c @ Y_c.T
    
    # Frobenius norm
    num = np.sum(C_XX * C_YY)
    den1 = np.sqrt(np.sum(C_XX * C_XX))
    den2 = np.sqrt(np.sum(C_YY * C_YY))
    
    if den1 == 0 or den2 == 0:
        return 0.0
    return num / (den1 * den2)
    
def recall_at_k(X, Y, k=1):
    """
    Computes Cross-Modal Instance Retrieval Recall@K using Cosine Similarity.
    X -> Y Retrieval Task.
    """
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    
    sim_matrix = X_norm @ Y_norm.T # (N, N)
    
    correct = 0
    N = X.shape[0]
    for i in range(N):
        top_k_indices = np.argsort(sim_matrix[i])[::-1][:k]
        if i in top_k_indices:
            correct += 1
            
    return correct / N

def calculate_advanced_modality_metrics(X_raw):
    a, t, v = X_raw[0], X_raw[1], X_raw[2]
    metrics = {}
    
    metrics['CKA (Audio-Text)'] = linear_cka(a, t)
    metrics['CKA (Text-Visual)'] = linear_cka(t, v)
    metrics['CKA (Visual-Audio)'] = linear_cka(v, a)
    
    metrics['Recall@1 (Audio -> Text)'] = recall_at_k(a, t, k=1)
    metrics['Recall@1 (Text -> Visual)'] = recall_at_k(t, v, k=1)
    metrics['Recall@1 (Visual -> Audio)'] = recall_at_k(v, a, k=1)
    
    return metrics
