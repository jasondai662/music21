# ------------------------------------------------------------------------------
# Name:         toolkit.py
# Purpose:      High-level analysis pipeline for common music analytics tasks.
#
# Authors:      OpenAI (AI-assisted)
#
# Copyright:    Copyright © 2026 Michael Scott Asato Cuthbert
# License:      BSD, see license.txt
# ------------------------------------------------------------------------------
'''
High-level analysis utilities for multi-task music analysis.

AI-assisted: This module was generated with assistance from an AI tool.
'''
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import argparse
import csv
import json
import pathlib
import re
import sys
import typing as t

import joblib  # type: ignore[import-untyped]

from music21 import chord
from music21 import converter
from music21 import corpus
from music21 import environment
from music21 import exceptions21
from music21 import harmony
from music21 import meter
from music21 import note
from music21 import stream
from music21.analysis import reduceChords
from music21.analysis import segmentByRests

if t.TYPE_CHECKING:
    from music21 import key

environLocal = environment.Environment('analysis.toolkit')

TaskName = t.Literal['key', 'harmony', 'melody', 'rhythm', 'structure']
SUPPORTED_TASKS: tuple[TaskName, ...] = (
    'key',
    'harmony',
    'melody',
    'rhythm',
    'structure',
)


class AnalysisToolkitException(exceptions21.Music21Exception):
    pass


@dataclass(frozen=True)
class AnalysisConfig:
    tasks: tuple[TaskName, ...] = SUPPORTED_TASKS
    audience: str = 'research'
    input_format: str | None = None
    source_type: t.Literal['file', 'corpus', 'stream'] = 'file'
    expand_repeats: bool = True
    make_measures: bool = True
    quantize: bool = True
    quantize_divisors: tuple[int, ...] = (4, 3)
    chord_reduction_max_chords: int = 3
    motif_length: int = 3
    motif_min_count: int = 2
    melody_part_strategy: t.Literal['highest', 'first'] = 'highest'


@dataclass(frozen=True)
class ChordEvent:
    offset: float
    measure_number: int | None
    offset_in_measure: float
    duration: float
    chord: chord.Chord


@dataclass
class AnalysisResult:
    source: str
    profile: dict[str, t.Any]
    metadata: dict[str, t.Any]
    key: dict[str, t.Any] | None
    harmony: dict[str, t.Any] | None
    melody: dict[str, t.Any] | None
    rhythm: dict[str, t.Any] | None
    structure: dict[str, t.Any] | None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, t.Any]:
        return {
            'source': self.source,
            'profile': self.profile,
            'metadata': self.metadata,
            'key': self.key,
            'harmony': self.harmony,
            'melody': self.melody,
            'rhythm': self.rhythm,
            'structure': self.structure,
            'warnings': self.warnings,
        }

    def to_summary_row(self) -> dict[str, str]:
        key_name = None
        if self.key:
            key_name = self.key.get('name')
        return {
            'source': self.source,
            'title': str(self.metadata.get('title') or ''),
            'composer': str(self.metadata.get('composer') or ''),
            'key': str(key_name or ''),
            'mode': str(self.key.get('mode') if self.key else ''),
            'segments': str(self.structure.get('segment_count') if self.structure else ''),
            'warnings': '; '.join(self.warnings),
        }


class AnalysisPipeline:
    def __init__(self, config: AnalysisConfig | None = None):
        self.config = config or AnalysisConfig()
        self._validate_tasks()
        self._cache: dict[str, AnalysisResult] = {}

    def analyze(self, source: str | stream.Stream, use_cache: bool = True) -> AnalysisResult:
        source_key = self._source_key(source)
        if use_cache and source_key in self._cache:
            return self._cache[source_key]

        score = self._load_score(source)
        score = self._normalize_score(score)

        warnings: list[str] = []
        metadata = _extract_metadata(score)
        profile = {
            'tasks': list(self.config.tasks),
            'audience': self.config.audience,
            'input_format': self.config.input_format,
            'source_type': self.config.source_type,
            'cleaning': {
                'expand_repeats': self.config.expand_repeats,
                'make_measures': self.config.make_measures,
                'quantize': self.config.quantize,
                'quantize_divisors': self.config.quantize_divisors,
            },
        }

        key_data = None
        harmony_data = None
        melody_data = None
        rhythm_data = None
        structure_data = None

        if 'key' in self.config.tasks:
            key_data = _analyze_key(score, warnings)
        if 'harmony' in self.config.tasks:
            harmony_data = _analyze_harmony(score, self.config, warnings)
        if 'melody' in self.config.tasks:
            melody_data = _analyze_melody(score, self.config, warnings)
        if 'rhythm' in self.config.tasks:
            rhythm_data = _analyze_rhythm(score, warnings)
        if 'structure' in self.config.tasks:
            structure_data = _analyze_structure(score, self.config, warnings)

        result = AnalysisResult(
            source=source_key,
            profile=profile,
            metadata=metadata,
            key=key_data,
            harmony=harmony_data,
            melody=melody_data,
            rhythm=rhythm_data,
            structure=structure_data,
            warnings=warnings,
        )
        if use_cache:
            self._cache[source_key] = result
        return result

    def batch_analyze(
        self,
        sources: Sequence[str],
        jobs: int | None = None,
        use_cache: bool = True,
    ) -> list[AnalysisResult]:
        '''
        Analyze multiple sources. Cache is only used for sequential execution.
        '''
        if jobs is not None and jobs < 1:
            raise AnalysisToolkitException('jobs must be >= 1')
        if jobs and jobs != 1:
            return joblib.Parallel(n_jobs=jobs)(
                joblib.delayed(_analyze_source)(source, self.config) for source in sources
            )
        return [self.analyze(source, use_cache=use_cache) for source in sources]

    def build_annotation_score(self, source: str | stream.Stream) -> stream.Score:
        score = self._load_score(source)
        score = self._normalize_score(score)
        reduction = _reduce_chords(score, self.config)
        events = _collect_chord_events(_reduction_part(reduction))
        return _annotate_with_chords(score, events)

    def _load_score(self, source: str | stream.Stream) -> stream.Score:
        if isinstance(source, stream.Stream):
            return _ensure_score(source)
        if self.config.source_type == 'corpus':
            parsed = corpus.parse(source)
        else:
            parsed = converter.parse(source, format=self.config.input_format)
        if isinstance(parsed, stream.Opus):
            if not parsed.scores:
                raise AnalysisToolkitException('No scores found in opus source')
            return _ensure_score(parsed.scores[0])
        return _ensure_score(parsed)

    def _normalize_score(self, score: stream.Score) -> stream.Score:
        working = score
        if self.config.expand_repeats:
            try:
                working = working.expandRepeats()
            except exceptions21.Music21Exception as exc:
                environLocal.printDebug(['expandRepeats failed', exc])
        if self.config.make_measures:
            working = working.makeMeasures(inPlace=False)
        if self.config.quantize:
            working = working.quantize(
                quarterLengthDivisors=self.config.quantize_divisors,
                recurse=True,
                inPlace=False,
            )
        return _ensure_score(working)

    def _validate_tasks(self) -> None:
        invalid = [task for task in self.config.tasks if task not in SUPPORTED_TASKS]
        if invalid:
            raise AnalysisToolkitException(f'Unsupported tasks: {invalid}')

    @staticmethod
    def _source_key(source: str | stream.Stream) -> str:
        if isinstance(source, stream.Stream):
            title = None
            if source.metadata:
                title = source.metadata.title
            return str(title or 'stream')
        return str(source)


def _analyze_source(source: str, config: AnalysisConfig) -> AnalysisResult:
    return AnalysisPipeline(config).analyze(source, use_cache=False)


def _ensure_score(stream_in: stream.Stream) -> stream.Score:
    if isinstance(stream_in, stream.Score):
        return stream_in
    score = stream.Score()
    if isinstance(stream_in, stream.Part):
        score.insert(0, stream_in)
        return score
    part = stream.Part()
    part.mergeElements(stream_in)
    score.insert(0, part)
    return score


def _extract_metadata(score: stream.Score) -> dict[str, t.Any]:
    metadata = score.metadata
    return {
        'title': metadata.title if metadata else None,
        'composer': metadata.composer if metadata else None,
        'parts': len(score.parts),
        'measures': len(score.recurse().getElementsByClass(stream.Measure)),
    }


def _analyze_key(score: stream.Score, warnings: list[str]) -> dict[str, t.Any] | None:
    try:
        analyzed_key: key.Key = score.analyze('key')
    except exceptions21.Music21Exception as exc:
        warnings.append(f'key analysis failed: {exc}')
        return None
    if analyzed_key is None:
        warnings.append('key analysis returned no result (insufficient pitch data)')
        return None
    return {
        'name': analyzed_key.tonicPitchNameWithCase,
        'mode': analyzed_key.mode,
        'correlationCoefficient': analyzed_key.correlationCoefficient,
        'tonalCertainty': analyzed_key.tonalCertainty()
        if analyzed_key.correlationCoefficient is not None
        else None,
    }


def _reduce_chords(score: stream.Score, config: AnalysisConfig) -> stream.Score:
    reduction = reduceChords.ChordReducer().run(
        score,
        maximumNumberOfChords=config.chord_reduction_max_chords,
    )
    return reduction


def _reduction_part(reduction: stream.Score) -> stream.Part:
    if reduction.parts:
        return reduction.parts[0]
    part = stream.Part()
    for element in reduction.recurse():
        part.append(element)
    return part


def _collect_chord_events(reduction_part: stream.Part) -> list[ChordEvent]:
    events: list[ChordEvent] = []
    for chord_obj in reduction_part.recurse().getElementsByClass(chord.Chord):
        measure = chord_obj.getContextByClass(stream.Measure)
        measure_number = measure.number if measure else None
        if measure is not None:
            offset_in_measure = float(chord_obj.getOffsetInHierarchy(measure))
        else:
            offset_in_measure = float(chord_obj.offset)
        offset = float(chord_obj.getOffsetInHierarchy(reduction_part))
        duration = float(chord_obj.duration.quarterLength)
        events.append(ChordEvent(offset, measure_number, offset_in_measure, duration, chord_obj))
    return events


def _serialize_chord_events(events: Iterable[ChordEvent]) -> list[dict[str, t.Any]]:
    serialized: list[dict[str, t.Any]] = []
    for event in events:
        root = event.chord.root()
        serialized.append({
            'offset': event.offset,
            'measure': event.measure_number,
            'offsetInMeasure': event.offset_in_measure,
            'duration': event.duration,
            'root': root.name if root else None,
            'quality': event.chord.quality,
            'commonName': event.chord.commonName,
            'pitches': [p.nameWithOctave for p in event.chord.pitches],
        })
    return serialized


def _analyze_harmony(
    score: stream.Score,
    config: AnalysisConfig,
    warnings: list[str],
) -> dict[str, t.Any] | None:
    try:
        reduction = _reduce_chords(score, config)
    except exceptions21.Music21Exception as exc:
        warnings.append(f'harmony reduction failed: {exc}')
        return None

    events = _collect_chord_events(_reduction_part(reduction))
    if not events:
        warnings.append('harmony reduction produced no chords')
        return None

    qualities = Counter(event.chord.quality for event in events)
    return {
        'eventCount': len(events),
        'qualityHistogram': dict(qualities),
        'events': _serialize_chord_events(events),
    }


def _select_melody_part(score: stream.Score, strategy: str) -> stream.Stream:
    if not score.parts:
        return score
    if strategy == 'first':
        return score.parts[0]
    best_part = None
    best_pitch = None
    for part in score.parts:
        pitches = [n.pitch.ps for n in part.recurse().getElementsByClass(note.Note)]
        if not pitches:
            continue
        avg_pitch = sum(pitches) / len(pitches)
        if best_pitch is None or avg_pitch > best_pitch:
            best_pitch = avg_pitch
            best_part = part
    return best_part or score.parts[0]


def _analyze_melody(
    score: stream.Score,
    config: AnalysisConfig,
    warnings: list[str],
) -> dict[str, t.Any] | None:
    melody_stream = _select_melody_part(score, config.melody_part_strategy)
    notes = list(melody_stream.recurse().getElementsByClass(note.Note))
    if not notes:
        warnings.append('melody analysis found no notes')
        return None

    pitch_values = [n.pitch.ps for n in notes]
    pitch_range = (min(pitch_values), max(pitch_values))
    intervals = segmentByRests.Segmenter.getIntervalList(melody_stream)
    interval_names = [i.directedName for i in intervals]
    motifs = _extract_interval_motifs(
        interval_names,
        config.motif_length,
        config.motif_min_count,
    )
    return {
        'noteCount': len(notes),
        'pitchRange': pitch_range,
        'intervalCount': len(intervals),
        'motifs': motifs,
    }


def _extract_interval_motifs(
    interval_names: Sequence[str],
    motif_length: int,
    min_count: int,
) -> list[dict[str, t.Any]]:
    if motif_length < 1:
        return []
    if len(interval_names) < motif_length:
        return []
    motif_counter: Counter[tuple[str, ...]] = Counter()
    for idx in range(len(interval_names) - motif_length + 1):
        motif_counter[tuple(interval_names[idx:idx + motif_length])] += 1
    motifs: list[dict[str, t.Any]] = [
        {'pattern': list(pattern), 'count': count}
        for pattern, count in motif_counter.items()
        if count >= min_count
    ]
    motifs.sort(key=lambda item: (-item['count'], item['pattern']))
    return motifs


def _analyze_rhythm(score: stream.Score, warnings: list[str]) -> dict[str, t.Any] | None:
    notes_and_rests = list(score.recurse().notesAndRests)
    if not notes_and_rests:
        warnings.append('rhythm analysis found no notes or rests')
        return None

    durations = Counter(element.duration.quarterLength for element in notes_and_rests)
    time_sigs = Counter(
        ts.ratioString for ts in score.recurse().getElementsByClass(meter.TimeSignature)
    )
    measures = list(score.recurse().getElementsByClass(stream.Measure))
    measure_count = len(measures)
    note_count = len(score.recurse().getElementsByClass(note.Note))
    notes_per_measure = None
    if measure_count:
        notes_per_measure = note_count / measure_count
    return {
        'noteCount': note_count,
        'measureCount': measure_count,
        'notesPerMeasure': notes_per_measure,
        'durationHistogram': dict(durations),
        'timeSignatureHistogram': dict(time_sigs),
    }


def _analyze_structure(
    score: stream.Score,
    config: AnalysisConfig,
    warnings: list[str],
) -> dict[str, t.Any] | None:
    melody_stream = _select_melody_part(score, config.melody_part_strategy)
    segments = segmentByRests.Segmenter.getSegmentsList(melody_stream)
    if not segments:
        warnings.append('structure analysis found no segments')
        return None
    segment_summaries = []
    empty_segments = 0
    for segment in segments:
        if not segment:
            empty_segments += 1
            continue
        start_note = segment[0]
        end_note = segment[-1]
        start_measure = start_note.getContextByClass(stream.Measure)
        end_measure = end_note.getContextByClass(stream.Measure)
        segment_summaries.append({
            'startOffset': start_note.offset,
            'endOffset': end_note.offset + end_note.duration.quarterLength,
            'startMeasure': start_measure.number if start_measure else None,
            'endMeasure': end_measure.number if end_measure else None,
            'noteCount': len(segment),
        })
    if empty_segments:
        warnings.append(f'structure analysis skipped {empty_segments} empty segments')
    return {
        'segmentCount': len(segment_summaries),
        'segments': segment_summaries,
    }


def _annotate_with_chords(score: stream.Score, events: Iterable[ChordEvent]) -> stream.Score:
    analysis_score = score.coreCopyAsDerivation('analysis')
    analysis_part = stream.Part(id='analysis')
    for event in events:
        symbol = harmony.chordSymbolFromChord(event.chord)
        analysis_part.insert(event.offset, symbol)
    analysis_score.insert(0, analysis_part)
    return analysis_score


def _source_stem(source: t.Any) -> str:
    source_str = str(source)
    path = pathlib.Path(source_str)
    if path.exists():
        return path.stem
    sanitized = re.sub(r'[^A-Za-z0-9_-]+', '_', source_str)
    sanitized = sanitized.strip('._')
    if sanitized in ('', '.', '..'):
        return 'analysis'
    return sanitized


def _write_json(results: Sequence[AnalysisResult], output: t.TextIO | None) -> None:
    payload = [result.to_dict() for result in results]
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    if output is None:
        print(data)
        return
    output.write(data)


def _write_csv(results: Sequence[AnalysisResult], output: t.TextIO | None) -> None:
    rows = [result.to_summary_row() for result in results]
    if not rows:
        return
    if output is None:
        output = sys.stdout
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def _parse_tasks(raw_tasks: str | None) -> tuple[TaskName, ...]:
    if not raw_tasks:
        return SUPPORTED_TASKS
    tasks = tuple(task.strip() for task in raw_tasks.split(',') if task.strip())
    invalid = [task for task in tasks if task not in SUPPORTED_TASKS]
    if invalid:
        raise AnalysisToolkitException(f'Unsupported tasks: {invalid}')
    return t.cast(tuple[TaskName, ...], tasks)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run multi-task music analysis.')
    parser.add_argument('sources', nargs='+', help='Input files or corpus ids.')
    parser.add_argument(
        '--corpus',
        action='store_true',
        help='Treat sources as music21 corpus ids.',
    )
    parser.add_argument('--format', dest='input_format', help='Input format override.')
    parser.add_argument(
        '--tasks',
        help='Comma-separated tasks (key,harmony,melody,rhythm,structure).',
    )
    parser.add_argument('--audience', default='research', help='Target audience label.')
    parser.add_argument('--output', help='Output file path.')
    parser.add_argument(
        '--output-format',
        choices=('json', 'csv'),
        default='json',
        help='Output format.',
    )
    parser.add_argument('--jobs', type=int, default=1, help='Parallel jobs.')
    parser.add_argument('--no-quantize', action='store_true', help='Disable quantization.')
    parser.add_argument('--no-expand-repeats', action='store_true', help='Disable repeats.')
    parser.add_argument('--no-make-measures', action='store_true', help='Disable measure making.')
    parser.add_argument('--motif-length', type=int, default=3, help='Motif length.')
    parser.add_argument('--motif-min-count', type=int, default=2, help='Min motif count.')
    parser.add_argument('--chord-max', type=int, default=3, help='Max chords per measure.')
    parser.add_argument(
        '--melody-part',
        choices=('highest', 'first'),
        default='highest',
        help='Melody part selection strategy.',
    )
    parser.add_argument(
        '--annotate',
        help='Directory to write annotated MusicXML exports.',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    tasks = _parse_tasks(args.tasks)
    config = AnalysisConfig(
        tasks=tasks,
        audience=args.audience,
        input_format=args.input_format,
        source_type='corpus' if args.corpus else 'file',
        expand_repeats=not args.no_expand_repeats,
        make_measures=not args.no_make_measures,
        quantize=not args.no_quantize,
        chord_reduction_max_chords=args.chord_max,
        motif_length=args.motif_length,
        motif_min_count=args.motif_min_count,
        melody_part_strategy=args.melody_part,
    )
    pipeline = AnalysisPipeline(config)
    results = pipeline.batch_analyze(args.sources, jobs=args.jobs)

    output_path = pathlib.Path(args.output) if args.output else None
    if output_path:
        with output_path.open('w', encoding='utf-8', newline='') as output_handle:
            if args.output_format == 'json':
                _write_json(results, output_handle)
            else:
                _write_csv(results, output_handle)
    else:
        if args.output_format == 'json':
            _write_json(results, None)
        else:
            _write_csv(results, None)

    if args.annotate:
        annotation_dir = pathlib.Path(args.annotate)
        annotation_dir.mkdir(parents=True, exist_ok=True)
        for source in args.sources:
            annotated = pipeline.build_annotation_score(source)
            name = _source_stem(source)
            output_file = annotation_dir / f'{name}_analysis.musicxml'
            annotated.write('musicxml', fp=str(output_file))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
