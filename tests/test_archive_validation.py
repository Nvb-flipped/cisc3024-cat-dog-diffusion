import hashlib
import json
import zipfile
import pytest
from src.release.verify_archive import validate_members


def archive(tmp_path, members):
    path = tmp_path / 'fixture.zip'
    with zipfile.ZipFile(path, 'w') as bundle:
        for name, value in members.items():
            bundle.writestr(name, value)
        bundle.writestr('GIT_HISTORY.txt', 'synthetic fixture\n')
        bundle.writestr('ARTIFACT_HASHES.json', json.dumps({name: hashlib.sha256(value.encode()).hexdigest()
                                                          for name, value in members.items()}))
    return path


@pytest.mark.parametrize('name', ['../outside.txt', '/absolute.txt', 'C:/outside.txt',
                                 'data/course.jpg', 'submission/other.pt', 'src\\escape.py'])
def test_archive_rejects_forbidden_or_escaping_members(tmp_path, name):
    with pytest.raises(ValueError, match='member'):
        validate_members(archive(tmp_path, {name: 'fixture'}))


def test_archive_rejects_case_collisions(tmp_path):
    path = archive(tmp_path, {'src/example.py': 'fixture', 'src/EXAMPLE.py': 'fixture'})
    with pytest.raises(ValueError, match='case-colliding'):
        validate_members(path)


def test_archive_rejects_embedded_course_photo_in_text(tmp_path):
    path = archive(tmp_path, {'figures/photo.drawio': '<mxfile>data:image/png;base64,fixture</mxfile>'})
    with pytest.raises(ValueError, match='Embedded image'):
        validate_members(path)


def test_archive_rejects_modified_member_hash(tmp_path):
    path = archive(tmp_path, {'README.md': 'fixture'})
    with zipfile.ZipFile(path, 'a') as bundle:
        bundle.writestr('src/unlisted.py', 'fixture')
    with pytest.raises(ValueError, match='hash inventory'):
        validate_members(path)
