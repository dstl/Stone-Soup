from os.path import join
import re

from setuptools_scm.utils import data_from_mime, trace
from setuptools_scm.version import meta, tags_to_versions


tag_re = re.compile(r'(?<=\btag: )([^,]+)\b')


def archival_to_version(data):
    trace('data', data)
    versions = tags_to_versions(tag_re.findall(data.get('ref-names', '')))
    if versions:
        return meta(versions[0])


def parse(root):
    archival = join(root, '.git_archival.txt')
    data = data_from_mime(archival)
    return archival_to_version(data)
