import streamlit as st
from pyaspeller import YandexSpeller
from io import StringIO
import torch
from streamlit_autorefresh import st_autorefresh
from foo import *


def process_json(file_contents):
    try:
        data = json.loads(file_contents)

        return json.dumps(data, indent=4)
    except Exception as e:
        return f"Ошибка обработки файла: {e}"


def count_words(text):
    words = text.split()
    return len(words)

def main():

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, T5ForConditionalGeneration, AutoModel
    except Exception as e:
        #st.error(f"Ошибка: {e}")
        st_autorefresh(interval=5000)

    st.title("Веб-приложение для анализа статей Знание.Вики")

    event = st.radio(
        "Выберите способ ввода данных о статье:",
        ('JSON-файл', 'URL', 'Новая статья')
    )

    uploaded_file = None
    url = ''
    title_new = ''
    string_data_new = ''

    if event == 'JSON-файл':
        uploaded_file = st.file_uploader("Загрузите файл JSON", type=["json"])
    elif event == 'Новая статья':
        title_new = st.text_input(label="Введите заголовок статьи")
        # Поле для ввода многострочного текста
        string_data_new = st.text_area(label='Введите текст')
    else:
        url = st.text_input(label="Введите URL статьи")



    if uploaded_file is not None or url != '' or (title_new !='' and string_data_new !=''):
        if uploaded_file is not None:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            string_data = stringio.read()
            st.code(string_data)
            string_data, table = extract_content_and_remove_from_string(string_data)
            outer_links, inner_links, string_data = clean_json(string_data)
            out_links = []
            for string in outer_links:
                start_index = string.find("url=")
                end_index = string.find("|", start_index)

                if start_index != -1 and end_index != -1:
                    substring = string[start_index + len("url="):end_index]
                    out_links.append(substring)

            title, string_data, images, type_text = parse_json_string(string_data)
            string_data = string_data.replace("}", "")
            string_data = string_data.replace("{", "")
            string_data = process_text_only(string_data)   # text


        elif url != '':
            json_data = extract_data_from_page(url)
            st.code(json_data)
            title, string_data, images, out_links = parse_json_string_from_url(json_data)

            title = title.split(' - ')[0]             # title
            title = title.split(' — ')[0]
            type_text = ["znanierussia.ru/articles"]
            #liststr = string_data.split('\n')
            #string_data = '\n'.join(liststr[4:])
            string_data = string_data.replace('\n', '')
            string_data = string_data.replace('\t', '')
            inner_links = string_data.count("[")                 # inner_links
            pattern = re.compile(r'\[\w+\\]')

            string_data = re.sub(pattern, '', string_data)           # text
            string_data = string_data.replace("}", "")
            string_data = string_data.replace("{", "")

        elif (title_new !='' and string_data_new !=''):
            # Поле для ввода заголовка
            st.code(f'{title_new}: {string_data_new}')

            title = title_new
            # Поле для ввода многострочного текста
            string_data = string_data_new
            inner_links = 0
            images = []
            out_links = []
            type_text = ["new/articles"]

        st.write("Статья успешно загружена!")

        option = st.selectbox("Выберите функцию обработки",
                              ("Просмотр содержимого",
                               "Анализ деструктивного контента",
                               "Выявление ненормативной лексики",
                               "Выявление и исправление ошибок",
                               "Проверка ссылок на источники иноагентов",
                               "Проверка соответствия заголовка содержанию",
                               "Определение соотношения полезного контента",
                               ))


        if option == "Просмотр содержимого":
            st.subheader("Содержимое файла:")
            st.write("I. Заголовок статьи:")
            st.code(title)
            st.write("II. Текст статьи:")
            st.code(string_data)
            st.write("III. Список ссылок на внешние источники:")
            selected = st.selectbox("Выберите ссылку:", out_links)


            st.write(f"Вы выбрали: {selected}")
            st.write(f"IV. Количество ссылок на статьи Знание.Вики: {inner_links}")
            st.write("V. Список ссылок на изображения:")
            #st.code('\n'.join(images))

            selected_item = st.selectbox("Выберите изображение:", images)


            st.write(f"Вы выбрали: {selected_item}")
            st.write(f"VI. {type_text}")

        elif option == "Выявление ненормативной лексики":


            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            import nltk
            import os

            def download_nltk_resources():
                resources = {
                    'punkt': 'tokenizers/punkt',
                    'wordnet': 'corpora/wordnet',
                    'stopwords': 'corpora/stopwords'
                }

                for resource, path in resources.items():
                    if not os.path.exists(os.path.join(nltk.data.find(''), path)):
                        nltk.download(resource)

            download_nltk_resources()

            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words('russian'))

            def read_rudewords(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    words = [line.strip() for line in file]
                return words

            mats = read_rudewords('rudewords.txt')
            destr_ = read_rudewords('destr.txt')

            def remove_punctuation(text):
                cleaned_text = re.sub(r'[^\w\s]', ' ', text)
                return cleaned_text

            def lemmatize_text_rude(text):
                words = nltk.word_tokenize(text)
                lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
                return ' '.join(lemmatized_words)

            def highlight_words_rude(text, mats, destr_, red_threshold=0.2, yellow_threshold=0.7):

                words = re.findall(r'\w+', text)

                filtered_words = [word for word in words if word.lower() not in stop_words and len(word) > 2]

                lemmatized_text = lemmatize_text_rude(' '.join(filtered_words))
                lemmatized_words = lemmatized_text.split()

                vectorizer = TfidfVectorizer(max_features=10000).fit(lemmatized_words + mats + destr_)
                words_vectors = vectorizer.transform(lemmatized_words)
                mats_vectors = vectorizer.transform(mats)
                destr_vectors = vectorizer.transform(destr_)

                mats_similarities = cosine_similarity(words_vectors, mats_vectors)
                destr_similarities = cosine_similarity(words_vectors, destr_vectors)

                highlighted_text = text
                # Проходимся по каждому слову и применяем логику выделения

                for i, word in enumerate(filtered_words):
                    lemmatized_word = lemmatizer.lemmatize(word)
                    if lemmatized_word in lemmatized_words:
                        idx = lemmatized_words.index(lemmatized_word)
                        max_mat_similarity = np.max(mats_similarities[idx])
                        max_destr_similarity = np.max(destr_similarities[idx])

                        if max_mat_similarity >= red_threshold:
                            highlighted_text = re.sub(r'\b{}\b'.format(re.escape(word)),
                                                      f'<span style="color:red;">{word}</span>', highlighted_text,
                                                      flags=re.IGNORECASE)
                        elif max_destr_similarity >= yellow_threshold:
                            highlighted_text = re.sub(r'\b{}\b'.format(re.escape(word)),
                                                      f'<span style="color:blue;">{word}</span>', highlighted_text,
                                                      flags=re.IGNORECASE)

                return highlighted_text



            #highlighted_text = highlight_words_rude(string_data, mats, destr_)
            highlighted_text = highlight_words_rude(string_data, mats, destr_, )
            st.markdown(highlighted_text, unsafe_allow_html=True)

        elif option == 'Проверка соответствия заголовка содержанию':

            from huggingface_hub import hf_hub_download
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, T5ForConditionalGeneration

            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            def remove_words_after_dash(text):

                cleaned_text = re.sub(r'—.*', '', text)
                return cleaned_text.strip()

            title = remove_words_after_dash(title)
            # Ввод названия статьи
            article_title = title
            if article_title:
                st.write("Введенное название статьи:", article_title)


            repo_id = "Vlad1m/check_topic"  # замените на ваш репозиторий
            filename = "model.pth"  # замените на имя вашего файла .pth
            file_path = hf_hub_download(repo_id=repo_id, filename=filename)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_name = "IlyaGusev/rut5_base_headline_gen_telegram"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name, resume_download=True)
            model.load_state_dict(torch.load(file_path, map_location=device))
            model.to(device)


            if 'model' not in st.session_state:
                st.session_state['model'] = model
            if 'tokenizer' not in st.session_state:
                st.session_state['tokenizer'] = tokenizer

            def generate_text(input_text, max_length):
                input_ids = st.session_state['tokenizer'].encode_plus(
                    input_text,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="pt"
                )["input_ids"].to(device)

                attention_mask = st.session_state['tokenizer'].encode_plus(
                    input_text,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="pt"
                )["attention_mask"].to(device)

                output = st.session_state['model'].generate(input_ids, attention_mask=attention_mask,
                                                            max_length=max_length)
                output_text = st.session_state['tokenizer'].decode(output[0], skip_special_tokens=True)
                return output_text

            with st.spinner():
                max_length = len(string_data) // 4
                pred_title = generate_text(string_data, max_length)
                pred_title2 = str(pred_title)

            def clean_string(s):
                s = re.sub(r'[^\w\s].*', '', s)
                s = re.sub('_.*', '', s)
                s = re.sub(r'\s+.*', '', s)
                return s

            cleaned_string = clean_string(pred_title2)

            st.write(f"Предполагаемое название статьи: {cleaned_string}")

            @st.cache_data
            def compare_texts(text1, text2):
                vectorizer = TfidfVectorizer().fit_transform([text1, text2])
                vectors = vectorizer.toarray()
                cos_sim = cosine_similarity(vectors)
                return cos_sim[0, 1]

            similarity = compare_texts(cleaned_string, article_title)

            if similarity >= 0.4:
                st.markdown('<span style="color:green;">Название соответствует содержанию статьи.</span>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:red;">Название не соответствует содержанию статьи.</span>',
                            unsafe_allow_html=True)
            placeholder = st.empty()
            placeholder.empty()

        elif option == "Анализ деструктивного контента":
            model = AutoModelForSequenceClassification.from_pretrained('Vlad1m/toxicity_analyzer')
            tokenizer = AutoTokenizer.from_pretrained('Vlad1m/toxicity_analyzer')

            def get_sentiment(text):
                with torch.no_grad():
                    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(
                        model.device)
                    proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()[0]

                return (model.config.id2label[proba.argmax()])  # proba


            st.markdown(':red[Результаты проверки на токсичность:]')
            st.write(get_sentiment(string_data))

            model = AutoModelForSequenceClassification.from_pretrained('Vlad1m/destractive_context')
            tokenizer = AutoTokenizer.from_pretrained('Vlad1m/destractive_context')
            st.markdown(':red[Результаты проверки на деструктивный контент:]')
            tox = get_sentiment(string_data)
            st.write(tox)


        elif option == 'Выявление и исправление ошибок':
            speller = YandexSpeller()
            data_correct = speller.spelled(string_data)

            if string_data == data_correct:
                st.write('Нет ошибок')
            else:
                st.markdown('<span style="color:red;">Статья содержит ошибки</span>', unsafe_allow_html=True)


                corrected_words = get_corrected_words(string_data, data_correct)
                mist = len(corrected_words)

                st.write(f'Количество исправленных слов: <span style="color:blue;">{mist}</span>',
                         unsafe_allow_html=True)
                st.write(f'Исправлены слова: <span style="color:blue;">{", ".join(corrected_words)}</span>',
                         unsafe_allow_html=True)
                st.text_area('Исправленный текст:', data_correct, height=300)


            mist_list = read_lines_from_file('Ling.txt')
            highlighted_text = highlight_words_in_text(string_data, mist_list, "red")
            st.markdown(highlighted_text, unsafe_allow_html=True)


        elif option == 'Проверка ссылок на источники иноагентов':

            import pandas as pd
            from natasha import (
                Segmenter,
                NewsEmbedding,
                NewsMorphTagger,
                NewsSyntaxParser,
                Doc,
                NewsNERTagger,
                MorphVocab
            )
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            from joblib import Parallel, delayed
            progress_bar = st.progress(0)

            segmenter = Segmenter()
            emb = NewsEmbedding()
            morph_tagger = NewsMorphTagger(emb)
            syntax_parser = NewsSyntaxParser(emb)
            ner_tagger = NewsNERTagger(emb)
            morph_vocab = MorphVocab()


            doc = Doc(string_data)
            doc.segment(segmenter)
            doc.tag_morph(morph_tagger)
            doc.parse_syntax(syntax_parser)
            if doc.sents:
                sent = doc.sents[0]
            else:
                print("В тексте не найдено ни одного предложения.")
            doc.tag_ner(ner_tagger)

            for token in doc.tokens:
                token.lemmatize(morph_vocab)

            for span in doc.spans:
                span.normalize(morph_vocab)

            vse_imena_v_text = {_.text: _.normal for _ in doc.spans}
            df3 = pd.DataFrame(list(vse_imena_v_text.items()), columns=['name', 'clean_name'])


            df3['clean_text'] = df3['name'].apply(lambda x: re.sub(r'[^\w\s]', ' ', x)).apply(
                lambda x: re.sub(r'\s+', ' ', x).strip())

            df4 = pd.read_csv('inoagenty.csv', sep='\t')
            df4['clean_text'] = df4['name'].apply(lambda x: re.sub(r'[^\w\s]', ' ', x)).apply(
                lambda x: re.sub(r'\s+', ' ', x).strip())


            vectorizer = TfidfVectorizer().fit(df4['clean_text'].tolist() + df3['clean_text'].tolist())

            def compare_texts(clean_name, clean_text):
                vectors = vectorizer.transform([clean_name, clean_text])
                cos_sim = cosine_similarity(vectors)
                return cos_sim[0, 1]

            def process_one_row(i):
                local_black = []
                text2 = df4['clean_text'][i]
                for j in range(len(df3)):
                    text1 = df3['clean_name'][j]
                    similarity = compare_texts(text1, text2)
                    if similarity >= 0.7:  # Повышение порога для большей точности
                        local_black.append(text1)
                return local_black


            results = Parallel(n_jobs=-1)(delayed(process_one_row)(i) for i in range(len(df4)))


            for i in range(len(df4)):
                progress_bar.progress((i + 1) / len(df4))

            black = [name for sublist in results for name in sublist]

            counter = len(black)
            st.write(f"Количество найденных отсылок на иноагентов: {counter}")

            for i, one in enumerate(black, 1):
                st.write(f"{i}. {one}")

            def highlight_words(text, phrases_to_highlight):
                phrases_to_highlight.sort(key=len, reverse=True)
                for phrase in phrases_to_highlight:
                    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                    text = pattern.sub(f"<span style='color:red;'>{phrase}</span>", text)
                return text

            black_phrases = [phrase for sublist in black for phrase in sublist.split()]

            highlighted_text = highlight_words(string_data, black_phrases)

            st.markdown(highlighted_text, unsafe_allow_html=True)


        elif option == 'Определение соотношения полезного контента':
            st.subheader("Материалы данной статьи:")
            st.write(f"Упомянуто ссылок на статьи znanierussia.ru: {inner_links}")
            st.write(f"Упомянуто ссылок на сторонние ресурсы: {len(out_links)}")
            st.write(f"Использовано изображений в статье: {len(images)}")

            def article_optimization(text, num_images, num_links):

                article_length = len(text.split())


                optimal_links = max(1, article_length // 300)
                optimal_images = max(1, article_length // 500)


                actual_links = min(optimal_links, num_links)
                actual_images = min(optimal_images, num_images)


                link_ratio = actual_links / article_length if article_length > 0 else 0
                image_ratio = actual_images / article_length if article_length > 0 else 0

                return link_ratio, image_ratio, article_length

            link_ratio, image_ratio, article_length = article_optimization(string_data, len(images), len(out_links))


            optimal_link_ratio = 1 / 300
            optimal_image_ratio = 1 / 500


            if article_length == 0:
                st.markdown("Статья не содержит текста.")
            elif link_ratio < optimal_link_ratio:
                st.markdown("Количество ссылок <span style=\"color:blue\">ниже</span> оптимального.",
                            unsafe_allow_html=True)
            elif abs(link_ratio - optimal_link_ratio) <= 0.01 * optimal_link_ratio:  # Допускаемая погрешность 1%
                st.markdown("Количество ссылок <span style=\"color:green\">оптимально</span>.", unsafe_allow_html=True)
            else:
                st.markdown("Количество ссылок <span style=\"color:red\">выше</span> оптимального.",
                            unsafe_allow_html=True)


            if article_length == 0:
                st.markdown("Статья не содержит текста.")
            elif image_ratio < optimal_image_ratio:
                st.markdown("Количество изображений <span style=\"color:blue\">ниже</span> оптимального.",
                            unsafe_allow_html=True)
            elif abs(image_ratio - optimal_image_ratio) <= 0.01 * optimal_image_ratio:  # Допускаемая погрешность 1%
                st.markdown("Количество изображений <span style=\"color:green\">оптимально</span>.",
                            unsafe_allow_html=True)
            else:
                st.markdown("Количество изображений <span style=\"color:red\">выше</span> оптимального.",
                            unsafe_allow_html=True)

    else:
        st.write("Загрузите файл, чтобы начать работу")

if __name__ == "__main__":
    main()