import csv

def page_titles_to_ids(titles, page_table_filename):
    mapping = {}
    titles_set = set(titles)
    with open(page_table_filename, encoding='utf8') as page_file:
        reader = csv.reader(page_file)
        for line in reader:
            id, namespace, title = int(line[0]), line[1], line[2]
            if namespace == '0' and title in titles_set:
                mapping[title] = id
    return mapping


def page_titles_for_categories(category_names, page2cat_filename):
    with open(page2cat_filename, encoding='utf8') as page2cat_file:
        reader = csv.reader(page2cat_file, delimiter='\t')
        titles = []
        for line in reader:
            title = line[0].replace(" ", "_")
            for category in [line[i] for i in range(1, len(line))]:
                if category in category_names:
                    titles.append(title)
                    break
        return titles


def load_redirects(page_table_filename, redirect_table_filename):
    target_title_to_source_id = {}
    with open(redirect_table_filename, encoding='utf8') as redirect_file:
        reader = csv.reader(redirect_file)
        for from_id, to_namespace, to_title, _, _ in reader:
            from_id = int(from_id)
            if to_namespace == '0':
                if to_title not in target_title_to_source_id:
                    target_title_to_source_id[to_title] = []
                target_title_to_source_id[to_title].append(from_id)

    source_to_target_id = {}
    with open(page_table_filename, encoding='utf8') as page_file:
        reader = csv.reader(page_file)
        for line in reader:
            id, title, is_redirect = int(line[0]), line[2], line[5]
            if is_redirect == '0': # Ignore double redirects
                for source_id in target_title_to_source_id.get(title, []):
                    source_to_target_id[source_id] = id

    return source_to_target_id


def links_between_pages(page_ids, pagelinks_table_filename, page_table_filename, redirect_table_filename):
    titles_linked_from = {}
    page_id_set = set(page_ids)
    with open(pagelinks_table_filename, encoding='utf8') as pagelinks_file:
        reader = csv.reader(pagelinks_file)
        for from_id, from_namespace, to_title, to_namespace in reader:
            from_id = int(from_id)
            if from_id in page_id_set and from_namespace == '0' and to_namespace == '0':
                if to_title not in titles_linked_from:
                    titles_linked_from[to_title] = []
                titles_linked_from[to_title].append(from_id)

    redirects = load_redirects(page_table_filename, redirect_table_filename)

    links = {id: [] for id in page_ids}
    with open(page_table_filename, encoding='utf8') as page_file:
        reader = csv.reader(page_file)
        for line in reader:
            id, title, is_redirect = int(line[0]), line[2], line[5]
            if is_redirect == '1':
                if id in redirects:
                    id = redirects[id]
                else:
                    continue

            for source_id in titles_linked_from.get(title, []):
                links[source_id].append(id)

    return titles_linked_from, redirects, links


def filter_for_main_namespace(input_filename, output_filename, field_indices):
    with open(input_filename, encoding='utf8') as input_file, \
         open(output_filename, mode='w+', encoding='utf8', newline='') as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file, quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            if all(row[idx] == '0' for idx in field_indices):
                writer.writerow(row)
