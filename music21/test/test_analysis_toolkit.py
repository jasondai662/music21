# ------------------------------------------------------------------------------
# Name:         test_analysis_toolkit.py
# Purpose:      Tests for analysis toolkit pipeline.
#
# Authors:      OpenAI (AI-assisted)
#
# Copyright:    Copyright © 2026 Michael Scott Asato Cuthbert
# License:      BSD, see license.txt
# ------------------------------------------------------------------------------
'''
Tests for the analysis toolkit pipeline.

AI-assisted: This test module was generated with assistance from an AI tool.
'''
from __future__ import annotations

import unittest

from music21 import converter
from music21 import stream
from music21.analysis import toolkit


class Test(unittest.TestCase):
    def test_basic_analysis(self):
        score = converter.parse('tinyNotation: 4/4 C4 D E F G A B c')
        pipeline = toolkit.AnalysisPipeline()
        result = pipeline.analyze(score)
        self.assertIsNotNone(result.key)
        self.assertIsNotNone(result.harmony)
        self.assertIsNotNone(result.melody)
        self.assertIsNotNone(result.rhythm)
        self.assertIsNotNone(result.structure)

    def test_summary_row(self):
        score = converter.parse('tinyNotation: 4/4 C4 D E F')
        result = toolkit.AnalysisPipeline().analyze(score)
        row = result.to_summary_row()
        self.assertIn('source', row)
        self.assertIn('title', row)

    def test_annotation_export(self):
        score = converter.parse('tinyNotation: 4/4 C4 E G C')
        pipeline = toolkit.AnalysisPipeline()
        annotated = pipeline.build_annotation_score(score)
        self.assertIsInstance(annotated, stream.Score)
        analysis_parts = [part for part in annotated.parts if part.id == 'analysis']
        self.assertTrue(analysis_parts)

    def test_batch_analyze_invalid_jobs(self):
        pipeline = toolkit.AnalysisPipeline()
        with self.assertRaises(toolkit.AnalysisToolkitException):
            pipeline.batch_analyze(['dummy'], jobs=0)
