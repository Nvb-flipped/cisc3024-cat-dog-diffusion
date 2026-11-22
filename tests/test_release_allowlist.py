import csv
import pytest
from src.release.package import allowed, selected_files
from src.release.readiness import audit_check


@pytest.mark.parametrize('name', ['data/student/index.csv', 'student/index.csv', '.git/config',
    'checkpoints/run.pt', 'submission/alternate.pt', 'submission/image.png', 'results/grid.png',
    'figures/example.svg', 'report/report.pdf', '.venvs/training/anything.py', '../secret.txt'])
def test_course_archive_excludes_prohibited_material(name):
    assert not allowed(name)


@pytest.mark.parametrize('name', ['submission/weights.pt', 'submission/generate.py', 'audit/audit.csv',
    'src/training/train.py', 'figures/F0/figure.drawio', 'figures/F0/source.csv',
    'presentation/presentation.pptx', 'presentation/presentation.pdf', 'README.md', 'demo.py',
    'requirements-demo.txt', 'EXPERIMENTS.md'])
def test_course_archive_retains_required_material(name):
    assert allowed(name)


def test_embedded_photos_are_not_smuggled_in_xml(tmp_path):
    figures = tmp_path / 'figures'
    figures.mkdir()
    (figures / 'photo.drawio').write_text('<mxfile>data:image/png;base64,fixture</mxfile>')
    (figures / 'vector.drawio').write_text('<mxfile><diagram/></mxfile>')
    assert [p.name for p in selected_files(tmp_path)] == ['vector.drawio']


def test_audit_completeness_requires_every_bad_reason(tmp_path):
    path = tmp_path / 'audit.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['verdict', 'reason'])
        writer.writeheader()
        writer.writerows([{'verdict': 'good', 'reason': ''} for _ in range(199)] + [{'verdict': 'bad', 'reason': ''}])
    with pytest.raises(ValueError, match='lacks reason'):
        audit_check(path)
    text = path.read_text()
    path.write_text(text.replace('bad,', 'bad,fixture reason'))
    assert audit_check(path)['rows'] == 200
