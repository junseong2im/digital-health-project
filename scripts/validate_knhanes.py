"""Validate original SPSS ZIPs without creating a study cohort or imputing data."""
from pathlib import Path
import hashlib
import json
import zipfile
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/knhanes'
OUT = ROOT / 'data/validation'
REQUESTED = 'age sex incm edu HE_BMI HE_wc HE_ht sm_presnt dr_month pa_aerobic HE_glu HE_sbp HE_dbp HE_TG HE_HDL_st2'.split()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reports = []
    for archive in sorted(RAW.glob('*.zip')):
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None, f'CRC failure: {archive.name}'
            members = [m for m in z.infolist() if m.filename.lower().endswith('.sav')]
            assert len(members) == 1, 'Expected exactly one SAV file'
            target = RAW / archive.stem / Path(members[0].filename).name
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(z.read(members[0]))
        df, meta = pyreadstat.read_sav(str(target), apply_value_formats=False)
        year = int('20' + archive.name[2:4])
        young = df.loc[df['age'].between(19, 39)]
        related = [c for c in df.columns if c.startswith(('DI1', 'DI2', 'DE1', 'HE_HDL', 'HE_fst', 'wt_', 'kstrata', 'psu'))]
        selected = list(dict.fromkeys([c for c in REQUESTED if c in df] + related))
        report = {
            'year': year, 'archive': str(archive.relative_to(ROOT)),
            'source': 'User supplied official download; official catalog verified separately',
            'archive_bytes': archive.stat().st_size,
            'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
            'sav_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
            'zip_crc_valid': True, 'spss_parse_valid': True,
            'rows': len(df), 'columns': len(df.columns),
            'age_19_39_before_exclusions': len(young),
            'requested_present': [c for c in REQUESTED if c in df],
            'requested_missing': [c for c in REQUESTED if c not in df],
            'variables': {c: {'label': meta.column_names_to_labels.get(c),
                'value_labels': meta.variable_value_labels.get(c, {}),
                'missing_young': int(young[c].isna().sum())} for c in selected},
            'note': 'Age counts are before diagnosis, medication, fasting, and missing-target exclusions. WHtR is not yet derived. No model trained.'
        }
        reports.append(report)
        (OUT / f'knhanes_{year}_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({k: report[k] for k in ['year','rows','columns','age_19_39_before_exclusions','requested_missing']}, ensure_ascii=False))
    (ROOT / 'data/knhanes_manifest.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
