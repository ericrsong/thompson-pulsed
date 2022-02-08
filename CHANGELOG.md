# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A changelog (this document!) roughly adhering to the standards specified in [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
- Parsing functions for ni_pcie7851r_ai
- Three_level experiment test scripts and instructions for how to run them
- Digital IQ demodulation capabilities for Time_Multitrace
- MT_Phasor and MT_Phase classes (inheriting methods from Time_Multitrace)

### Changed
- Parsing function names changed to give unique identification
- Default cavity resonance probe measurement uses an IQ demodulation strategy, as opposed to an FFT + curve fit

### Removed
- Time_Trace and Frequency_Trace (instead use Time_Multitrace and Frequency_Multitrace)

## [0.1.0] - 2022-02-01

### Added

- Version control using the setuptools_scm package

## [0.0.0] - 2022-01-25

### Added

- Barebones package structure for a pulsed experiment analysis package
- Class definitions for single traces, multitraces, and shots (multiple traces from a single data file)
- Parser module in /core/ containing a single parsing function for the NI PCI-5105 oscilloscope board
- Experiment-specific analysis code for the 3L experiment, stored in /expts/three_level.py

[Unreleased]: https://github.com/dylan-j-young/thompson-pulsed/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dylan-j-young/thompson-pulsed/compare/v0.0.0...v0.1.0
[0.0.0]: https://github.com/dylan-j-young/thompson-pulsed/tree/v0.0.0