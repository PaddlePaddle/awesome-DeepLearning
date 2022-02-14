"""
read map for model
"""
from reader.reader_utils import regist_reader, get_reader
import reader.tsminf_reader as tsminf_reader
import reader.audio_reader as audio_reader
import reader.bmninf_reader as bmninf_reader
import reader.feature_reader as feature_reader

# regist reader, sort by alphabet
regist_reader("TSM", tsminf_reader.TSMINFReader)
regist_reader("PPTSM", tsminf_reader.TSMINFReader)
regist_reader("AUDIO", audio_reader.AudioReader)
regist_reader("BMN", bmninf_reader.BMNINFReader)
regist_reader("ACTION", feature_reader.FeatureReader)
