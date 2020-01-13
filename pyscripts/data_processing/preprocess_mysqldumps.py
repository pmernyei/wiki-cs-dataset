"""
Preprocess the 'page', 'redirect' and 'pagelinks' table SQL dumps by turning
them into CSVs and filtering for only the entries relating to the default
namespace.
"""
import sys
import csv
import os

from mysqldump_to_csv import dump_to_csv

def filter_for_main_namespace(input_filename, output_filename, field_indices):
    """
    Given an input CSV and a set of field indices, output only those rows where
    the corresponding columns contain 0.
    Used to discard information not relating to the default Wikipedia namespace
    (e.g. talk or other meta pages).
    """
    with open(input_filename, encoding='utf8') as input_file, \
         open(output_filename, mode='w+',
                encoding='utf8', newline='') as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file, quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            if all(row[idx] == '0' for idx in field_indices):
                writer.writerow(row)

if __name__ == '__main__':
    # The directory containing the uncompressed datadumps
    dumps_dir = sys.argv[1]
    # E.g. 'enwiki-20190820'
    datadumps_prefix = sys.argv[2]

    # List the appropriate indices for each table. Note that the field order
    # in the pagelinks table contradicts documentation, fields 2 and 3 are
    # apparently the other way around.
    for table, filter_indices in [('page', [1]),
                                  ('redirect', [1]),
                                  ('pagelinks', [1, 3])]:
        source = os.path.join(dumps_dir, datadumps_prefix + '-' + table +'.sql')
        intermediate = os.path.join(dumps_dir, table + '-unfiltered-temp.csv')
        result = os.path.join(dumps_dir, table + '.csv')
        dump_to_csv(source, intermediate)
        filter_for_main_namespace(intermediate, result, filter_indices)
        os.remove(intermediate)
