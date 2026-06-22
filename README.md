# representational-behavioral-gap

Исследовательский репозиторий для анализа **разрыва между внутренними представлениями и поведением vision-language models (VLMs)** на задачах shape/texture bias.  
Проект сравнивает несколько VLM и отвечает на два основных вопроса:

1. **Что кодируется во внутренних языковых скрытых состояниях модели** при ответе на изображения из stylized ImageNet: shape или texture?
2. **Можно ли сдвигать поведение модели** с помощью activation steering, вычитая направления, найденные линейными пробами?

В текущем виде репозиторий содержит код и результаты для моделей:

- `llava`
- `paligemma2mix`
- `qwen3vl`

---

## Идея проекта

На stylized ImageNet форма объекта и его текстура намеренно расходятся.  
Например, изображение может иметь **форму кошки**, но **текстуру слона**. Это позволяет измерять:

- **shape bias** — склонность модели отвечать по форме;
- **texture bias** — склонность модели отвечать по текстуре.

В репозитории реализованы два связанных направления анализа:

### 1. Анализ представлений
Из моделей извлекаются скрытые состояния последнего токена ответа по всем декодерным слоям.  
Затем на каждом слое обучаются линейные классификаторы:

- для предсказания `shape`-метки,
- для предсказания `texture`-метки.

Это позволяет оценить, насколько хорошо shape и texture разделимы в представлениях модели на разных слоях.

### 2. Анализ поведения через activation steering
После обучения линейных проб берутся найденные направления для shape/texture и из последнего скрытого состояня вычитается проекция на это направление:

- вычитание `texture`-направления должно усиливать **shape bias**;
- вычитание `shape`-направления должно усиливать **texture bias**.

Далее измеряется, насколько реально меняются ответы модели.

---

## Структура репозитория

```text
.
├── environment.yml
├── scripts/
│   └── run_language_hidden_states_extraction.sh
├── src/
│   ├── experiments/
│   │   ├── activation_steering.py
│   │   ├── hidden_states_extraction.py
│   │   ├── linear_probing.py
│   │   └── plot_linear_classifiers_cross_val.py
│   └── draw/
│       ├── language_hidden_states_linear_probing_summary.py
│       └── plot_activation_steering_shape_bias.py
└── data/
    ├── activation_steering/
    ├── language_hidden_states/
    ├── linear_classifiers/
    ├── linear_classifiers_cross_val/
    ├── language_features/
    ├── vision_features/
    ├── vision_shape_texture_metrics/
    ├── paper_figures/
    └── stylized-imagenet/
```

### Основные директории с данными

- `data/stylized-imagenet/style-transfer-preprocessed-512/`  
  Stylized ImageNet, используемый как входной датасет.

- `data/language_hidden_states/`  
  Сохранённые скрытые состояния языковой части VLM по слоям (`.npy`) и агрегированные summary-файлы.

- `data/linear_classifiers/`  
  Веса линейных проб для shape/texture, обученных по скрытым состояниям.

- `data/activation_steering/`  
  Результаты steering-экспериментов и графики метрик по `alpha`.

- `data/paper_figures/`  
  Подготовленные итоговые иллюстрации.

---

## Используемый стек

Окружение описано в `environment.yml`.

Ключевые зависимости:

- Python 3.10
- PyTorch
- Transformers
- bitsandbytes
- NumPy
- scikit-learn
- matplotlib
- seaborn
- tqdm
- Pillow

---

## Установка окружения

### Через conda

```bash
conda env create -f environment.yml
conda activate transformers
```

---

## Внешние зависимости и ограничения

### Локальные веса моделей
Скрипты ожидают, что веса моделей уже скачаны локально и лежат в переменной окружения:

```python
export MODELS_ROOT_PATH="path/to/hugging/face/models"
```

Или в директории по-умолчанию:

```python
MODELS_ROOT_PATH = Path("data/models")
```


Он используется в:

- `src/experiments/hidden_states_extraction.py`
- `src/experiments/activation_steering.py`

Если у вас другой путь к моделям, его нужно изменить в этих файлах.

### Поддерживаемые модели
В коде настроены следующие конфигурации:

- `llava` → `llava-hf/llava-1.5-7b-hf`
- `paligemma2mix` → `google/paligemma2-10b-mix-224`
- `qwen3vl` → `Qwen/Qwen3-VL-8B-Instruct`

### Аппаратные требования
При наличии CUDA модели загружаются с:

- `load_in_8bit=True`
- `torch.float16`

Без GPU код тоже может работать, но существенно медленнее и, вероятно, не для больших моделей в реальной практике.

---

## Датасет и формат меток

Имена файлов в stylized ImageNet интерпретируются как:

```text
{shape_name}{id}-{texture_name}{id}.png
```

Примеры:

- `airplane1-bicycle2.png`
- `bear4-oven2.png`

Из имени извлекаются:

- `shape_name` — истинная форма;
- `texture_name` — истинная текстура.

Для VQA-модели используется фиксированный prompt с 16 классами, каждому сопоставлена буква:

- A — airplane
- B — bear
- C — bicycle
- D — bird
- E — boat
- F — bottle
- G — car
- H — cat
- I — chair
- J — clock
- K — dog
- L — elephant
- M — keyboard
- N — knife
- O — oven
- P — truck

Модель должна ответить **одной буквой**.

---

## Основные эксперименты

## 1. Извлечение hidden states

Скрипт: `src/experiments/hidden_states_extraction.py`

Что делает:

- загружает VLM и processor;
- прогоняет все `.png` из stylized ImageNet;
- извлекает скрытые состояния последнего токена ответа для **всех слоёв**;
- сохраняет результат в `.npy` для каждого изображения;
- дополнительно сохраняет `shape_bias.json` с поведенческой метрикой модели на выбранном prompt.

### Поддерживаемые prompt-режимы

- `default`
- `shape_biased`
- `texture_biased`

### Пример запуска

Из корня репозитория:

```bash
python src/experiments/hidden_states_extraction.py \
  --model llava \
  --output-dir data/language_hidden_states/llava \
  --prompt-type default
```

Пример для prompt-manipulation:

```bash
python src/experiments/hidden_states_extraction.py \
  --model qwen3vl \
  --output-dir data/language_hidden_states/qwen3vl/shape_biased_prompts \
  --prompt-type shape_biased
```

### Что сохраняется

- `data/language_hidden_states/<model>/**/*.npy` — массив формы `(num_layers, hidden_dim)`
- `shape_bias.json` — summary по shape/texture bias для соответствующего запуска

---

## 2. Линейное probing

Скрипт: `src/experiments/linear_probing.py`

Что делает:

- загружает сохранённые hidden states;
- для каждого слоя обучает два `LogisticRegression` классификатора:
  - shape classifier,
  - texture classifier;
- вычисляет `macro F1` на test split;
- сохраняет веса проб и графики качества по слоям.

### Пример запуска

```bash
python src/experiments/linear_probing.py \
  --input-dir data/language_hidden_states/llava \
  --output-dir data/linear_classifiers/llava \
  --random-state 42
```

### Multi-seed запуск

```bash
python src/experiments/linear_probing.py \
  --input-dir data/language_hidden_states/llava \
  --output-dir data/linear_classifiers_cross_val/llava \
  --random-states 1 2 3 4 5
```

### Что сохраняется

При single-seed запуске:

- `shape_ws.npy`
- `texture_ws.npy`
- `f1_metrics.json`
- `f1_metrics.png`

При multi-seed запуске:

- `shape_ws_by_seed.npy`
- `texture_ws_by_seed.npy`
- `f1_metrics.json`
- `f1_metrics.png`

---

## 3. Сводка по линейным пробам для разных prompt-режимов

Скрипт: `src/draw/language_hidden_states_linear_probing_summary.py`

Что делает:

- рекурсивно сканирует `data/language_hidden_states/`;
- находит директории с hidden states для разных моделей и prompt-режимов;
- для последних `N` слоёв обучает shape/texture probes;
- усредняет F1 и считает:
  - `delta_f1_texture_minus_shape = avg_texture_f1 - avg_shape_f1`

Положительное значение означает, что в последних слоях **texture легче линейно декодируется, чем shape**.

### Пример запуска

```bash
python src/draw/language_hidden_states_linear_probing_summary.py \
  --input-dir data/language_hidden_states \
  --output-md data/language_hidden_states/linear_probe_delta_f1_summary.md \
  --output-json data/language_hidden_states/linear_probe_delta_f1_summary.json \
  --last-n-layers 4 \
  --random-state 42
```

### Текущая summary-таблица

Файл: `data/language_hidden_states/linear_probe_delta_f1_summary.md`

| model | neutral | shape_biased | texture_biased |
| --- | ---: | ---: | ---: |
| llava | 0.22494 | 0.186076 | 0.248888 |
| paligemma2mix | 0.280515 | 0.200656 | 0.337166 |
| qwen3vl | 0.251593 | 0.196767 | 0.302644 |

Интерпретация:

- для всех трёх моделей `delta_f1_texture_minus_shape > 0`;
- даже при `shape_biased` prompt texture остаётся лучше линейно декодируемой в последних слоях;
- при `texture_biased` prompt разрыв обычно становится ещё больше.

---

## 4. Activation steering

Скрипт: `src/experiments/activation_steering.py`

Что делает:

1. Загружает модель.
2. Загружает обученные веса линейных проб:
   - `shape_ws.npy`
   - `texture_ws.npy`
3. Для каждого изображения получает последнее скрытое состояние последнего токена.
4. Строит patched-варианты:
   - вычитает проекцию на shape-направление;
   - вычитает проекцию на texture-направление.
5. Для диапазона `alpha` оценивает, как меняются ответы модели.
6. Сохраняет итоговые bias-метрики и график.

### Пример запуска

```bash
python src/experiments/activation_steering.py \
  --model llava \
  --alpha-start 0 \
  --alpha-end 30 \
  --alpha-step 0.5
```

### Выходные артефакты

Для каждой модели в `data/activation_steering/<model>/`:

- `answers.json` — метрики по всем значениям `alpha`
- `answers_metrics.png` — графики изменения bias/flip-rate/stability

### Основные метрики

Скрипт вычисляет:

- `shape_bias_orig`, `shape_bias_patched`
- `texture_bias_orig`, `texture_bias_patched`
- `shape_bias_gain`
- `texture_bias_gain`
- `flip_rate_tex_to_shape_pct`
- `flip_rate_shape_to_tex_pct`
- `stable_predictions_pct`

Идея интерпретации:

- если вычитание `texture`-компоненты увеличивает `shape_bias`, steering работает в ожидаемом направлении;
- если вычитание `shape`-компоненты увеличивает `texture_bias`, steering тоже работает в ожидаемом направлении;
- `stable_predictions_pct` показывает, насколько вмешательство вообще меняет поведение модели.

---

## Дополнительные скрипты

### `src/experiments/plot_linear_classifiers_cross_val.py`
Скрипт для визуализации результатов cross-validation / multi-seed экспериментов линейных классификаторов.

### `src/draw/plot_activation_steering_shape_bias.py`
Скрипт для построения итоговых графиков по activation steering.

---

## Готовые данные и артефакты в репозитории

В репозитории уже лежат промежуточные и итоговые результаты:

- hidden states для ряда моделей и prompt-режимов;
- обученные линейные классификаторы;
- activation steering outputs;
- summary markdown/json;
- итоговые картинки в `data/paper_figures/`.

Это позволяет:

- не запускать полный pipeline с нуля;
- использовать репозиторий как архив экспериментов;
- быстро строить дополнительные summary и figures поверх уже посчитанных данных.

---

## Рекомендуемый порядок воспроизведения

### Базовый pipeline

1. Извлечь hidden states:
   ```bash
   python src/experiments/hidden_states_extraction.py --model llava --output-dir data/language_hidden_states/llava --prompt-type default
   ```

2. Обучить линейные пробы:
   ```bash
   python src/experiments/linear_probing.py --input-dir data/language_hidden_states/llava --output-dir data/linear_classifiers/llava --random-state 42
   ```

3. Запустить activation steering:
   ```bash
   python src/experiments/activation_steering.py --model llava --alpha-start 0 --alpha-end 30 --alpha-step 0.5
   ```

4. Построить summary:
   ```bash
   python src/draw/language_hidden_states_linear_probing_summary.py
   ```

### Batch-запуск hidden state extraction
Есть вспомогательный shell-скрипт:

```bash
sh scripts/run_language_hidden_states_extraction.sh
```

Примечание: в текущем виде он вызывает `python hidden_states_extraction.py ...` без префикса `src/experiments/`, поэтому запускать его стоит либо после адаптации путей, либо из контекста, где этот файл доступен как `hidden_states_extraction.py`.

---

## Что важно учитывать

- Пути к локальным весам моделей сейчас не параметризованы через CLI или переменные окружения.
- Репозиторий ориентирован на локальный исследовательский запуск, а не на упакованный production workflow.
- Большая часть артефактов уже сохранена в `data/`, поэтому многие графики и таблицы можно использовать сразу.
- Для больших моделей желателен GPU с поддержкой CUDA и достаточным объёмом памяти.

---

## Возможные улучшения репозитория

- вынести `MODELS_ROOT_PATH` в аргумент командной строки или переменную окружения;
- добавить единый launcher / Makefile;
- унифицировать пути в shell-скриптах;
- добавить описание форматов `.npy` и `.json` артефактов;
- добавить раздел с ключевыми количественными выводами по activation steering.

---

## Краткий вывод по содержанию репозитория

Репозиторий показывает исследовательский pipeline, в котором:

- из VLM извлекаются скрытые языковые представления;
- проверяется, что в них линейно декодируется лучше — shape или texture;
- затем эти направления используются для causal-style intervention через activation steering;
- после этого измеряется, насколько внутренние направления действительно контролируют наблюдаемое поведение модели.

Именно этот разрыв между **декодируемостью признака в представлении** и **реальным поведенческим bias** и составляет центральную тему проекта.