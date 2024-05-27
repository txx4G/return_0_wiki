import requests
from bs4 import BeautifulSoup
import json, re
from difflib import ndiff

def update_file(key, new_value, filename):
    # Читаем содержимое файла
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Ищем строку с ключом
    for i, line in enumerate(lines):
        if line.startswith(key):
            # Обновляем строку с новым значением
            lines[i] = f'{key}: {new_value}\n'
            break

    # Записываем обновленное содержимое обратно в файл
    with open(filename, 'w') as file:
        file.writelines(lines)

# функции обработки URL

def highlight_word_in_text(text, word, color):
    # Инициализируем переменные для хранения результата и текущего индекса
    highlighted_text_ = ""
    current_index = 0

    # Пока не достигнут конец текста
    while current_index < len(text):
        # Ищем следующее вхождение слова
        next_index = text.find(word, current_index)

        # Если слово найдено
        if next_index != -1:
            # Добавляем часть текста до слова к результату
            highlighted_text_ += text[current_index:next_index]
            # Добавляем отформатированное слово к результата
            highlighted_text_ += f'<span style="color:{color}">{text[next_index:next_index + len(word)]}</span>'
            # Обновляем текущий индекс
            current_index = next_index + len(word)
        else:
            # Если слово не найдено, добавляем оставшийся текст к результату и завершаем цикл
            highlighted_text_ += text[current_index:]
            break

    return highlighted_text_


def highlight_words_in_text(text, word_list_, color):
    # Инициализируем текст результата
    highlighted_text_ = text

    # Проходим по каждому слову в списке
    for word in word_list_:
        # Вызываем функцию highlight_word_in_text для каждого слова
        highlighted_text_ = highlight_word_in_text(highlighted_text_, word, color)

    return highlighted_text_
#
def extract_data_from_page(url):
    # Выполняем GET-запрос к указанной странице
    response = requests.get(url)

    # Проверяем успешность запроса
    if response.status_code == 200:
        # Создаем объект BeautifulSoup для парсинга HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # Находим заголовок страницы
        title = soup.find('title').get_text()

        # Находим текст статьи
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])

        # Находим ссылки на картинки
        image_links = [img['src'] for img in soup.find_all('img')]

        # Находим ссылки на сторонние ресурсы
        external_links = [a['href'] for a in soup.find_all('a') if a.get('href') and a['href'].startswith('http')]

        # Создаем словарь для данных
        data = {
            #'url': url,
            'title': title,
            'text': text,
            'image_links': image_links,
            'external_links': external_links
        }

        # Возвращаем данные в формате JSON
        return json.dumps(data, ensure_ascii=False)
    else:
        # Если запрос не удался, возвращаем сообщение об ошибке
        return json.dumps({'error': 'Failed to fetch the page'}, ensure_ascii=False)


def parse_json_string_from_url(json_string):

    data = json.loads(json_string)
    #url = data.get('type', '')
    title = data.get('title', '')
    title = title.split(' - ')[0]
    text = data.get('text', '')
    images = data.get('image_links', [])
    links = data.get('external_links', [])

    text = text.replace('\xa0', '')

    bracketed_pattern = r'\[\[(.*?)\]\]'

    # Находим все подстроки, соответствующие паттерну
    matches = re.findall(bracketed_pattern, text)

    # Обрабатываем каждую подстроку
    for match in matches:
        # Если первое слово - "Файл", удаляем подстроку
        if match.startswith('Файл'):
            text = text.replace('[[' + match + ']]', ' ')
        # Если внутри есть символ '|', оставляем часть после него
        elif '|' in match:
            text = text.replace('[[' + match + ']]', match.split('|', 1)[-1] + ' ')
        # В противном случае оставляем содержимое, удаляя скобки
        else:
            text = text.replace('[[' + match + ']]', match+ ' ')


    pattern = re.compile(r'\[\w+\]')
    text = re.sub(pattern, '', text)
    return title, text, images, links  # url

# algos
def colorize_text(text, rude_words, destr_words):
    words = text.split()
    result = []

    for word in words:
        if word.lower() in destr_words:
            result.append('<span style="color: green;">{}</span>'.format(word))
        elif word.lower() in rude_words:
            result.append('<span style="color: red;">{}</span>'.format(word))
        else:
            result.append('<span style="color: blue;">{}</span>'.format(word))

    colored_text = ' '.join(result)

    return colored_text

def read_lines_from_file(file_path):
    """Читает строки из файла и возвращает список строк."""
    lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            lines.append(line.strip())
    return lines

def get_corrected_words(original_text, corrected_text):
    original_words = original_text.split()
    corrected_words = corrected_text.split()
    diff = list(ndiff(original_words, corrected_words))
    corrected_words = [word[2:] for word in diff if word.startswith('- ')]
    return corrected_words

# JSON processing

def clean_json(text):
    # Паттерн для поиска подстрок между угловыми скобками
    angle_brackets_pattern = r'<[^>]*>'
    # Удаляем подстроки, соответствующие паттерну
    clean_text = re.sub(angle_brackets_pattern, '', text)
    clean_text = clean_text.replace('\n', '')
    double_braces_pattern = r'{{(.*?)}}'
    # Находим все подстроки, соответствующие паттерну
    matches = re.findall(double_braces_pattern, clean_text)
    # Удаляем найденные подстроки из текста
    clean_text = re.sub(double_braces_pattern, '', clean_text)
    # Подсчитываем количество подстрок \xa0
    count = clean_text.count('\xa0')
    # Удаляем все вхождения подстроки \xa0 из текста
    clean_text = clean_text.replace('\xa0', '')
    return list(set(matches)), count, clean_text


def parse_json_string(json_string):

    data = json.loads(json_string)
    title = data.get('title', '')
    text = data.get('text', '')
    images = data.get('images', [])
    article_type = data.get('type', '')

    return title, text, images, article_type


def extract_content_and_remove_from_string(input_string):
    start_index = input_string.find('{{НП\n')
    end_index = input_string.find('}}\n\'\'\'', start_index)
    # Если найдена подстрока, извлекаем содержимое между скобками и удаляем подстроку из исходной строки
    if start_index != -1 and end_index != -1:
        content = input_string[start_index + 5:end_index]
        return input_string[:start_index] + input_string[end_index + 5:], content
    else:
        return input_string, None

# Text processing

def process_text_only(text):
    # Паттерн для выделения пунктов

    point_pattern = r'={2}([^=]+)={2}'
    # Паттерн для выделения подпунктов
    subpoint_pattern = r'={4}([^=]+)={4}'

    # Заменяем выделители на соответствующие символы форматирования
    text = re.sub(point_pattern, r'\n\t\1\n', text)
    text = re.sub(subpoint_pattern, r'\n\t\t\1\n', text)

    # Удаляем оставшиеся выделители
    text = text.replace('=', '')

    # Паттерн для поиска подстрок типа [[...]]
    bracketed_pattern = r'\[\[(.*?)\]\]'

    # Находим все подстроки, соответствующие паттерну
    matches = re.findall(bracketed_pattern, text)

    # Обрабатываем каждую подстроку
    for match in matches:
        # Если первое слово - "Файл", удаляем подстроку
        if match.startswith('Файл'):
            text = text.replace('[[' + match + ']]', '')
        # Если внутри есть символ '|', оставляем часть после него
        elif '|' in match:
            text = text.replace('[[' + match + ']]', match.split('|', 1)[-1])
        # В противном случае оставляем содержимое, удаляя скобки
        else:
            text = text.replace('[[' + match + ']]', match)

    # Удаляем символы "'", "'" и заменяем символ '#' на перенос строки
    text = text.replace("'", "").replace('"', "").replace("#", "\n")
    # После знака ';' ставим перенос строки
    text = text.replace(";", ";\n\t")
    text = text.replace("\n\n", "\n")
    text = text.replace("|", "")
    text = text.replace("— ", " — ")
    text = text.replace('\n','')
    text = text.replace('\t', '')
    phrase = 'Примечания'
    if text.endswith(phrase):
        # Если да, удаляем фразу из конца строки
        return text[: -len(phrase)].rstrip()
    else:
        # Если нет, возвращаем исходную строку без изменений
        return text

# links processing
def parse_marked_string(marked_string):
    # Разделитель меток
    delimiter = '|'
    # Возможные метки
    possible_labels = ['url', 'title', 'author', 'date', 'publisher', 'access-date']

    # Разделяем строку по разделителю '|'
    parts = marked_string.split(delimiter)
    # Создаем словарь для хранения меток и их значений
    parsed_data = {}

    # Обрабатываем каждую часть
    for part in parts:
        # Разделяем часть на метку и значение по первому знаку '='
        label_value_pair = part.split('=', 1)
        # Если метка присутствует в списке возможных меток
        if label_value_pair[0] in possible_labels:
            # Добавляем метку и её значение в словарь
            parsed_data[label_value_pair[0]] = label_value_pair[1] if len(label_value_pair) > 1 else ''

    return parsed_data


def parse_links(matches):
    links = []
    for x in matches:
        if x.startswith('Cite') or x.startswith('cite'):
            links.append(parse_marked_string(x))

    return links

