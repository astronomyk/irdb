'''Basic unit tests for irdb/METIS'''
# pylint: disable=no-self-use, missing-class-docstring
# pylint: disable=missing-function-docstring

import os
from glob import glob

import pytest

import numpy as np
#from astropy.io.fits import HDUList
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import scopesim
from scopesim.source.source_templates import star_field
#import scopesim_templates as sim_tp

PLOTS = False
PKGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
scopesim.rc.__config__["!SIM.file.local_packages_path"] = PKGS_DIR


class TestLoads:
    '''Test that irdb/METIS is loaded and an OpticalTrain is produced'''

    @pytest.mark.parametrize("themode",
                             ["img_lm", "img_n", "lss_l", "lss_m", "lss_n"])
    def test_scopesim_loads_package(self, themode):
        '''Load the configuration for all supported modes'''
        cmd = scopesim.UserCommands(use_instrument="METIS",
                                    set_modes=[themode])
        assert isinstance(cmd, scopesim.UserCommands)

        metis = scopesim.OpticalTrain(cmd)
        assert isinstance(metis, scopesim.OpticalTrain)


YAML_LIST = glob(os.path.join(PKGS_DIR, "METIS/*.yaml"))
@pytest.fixture(name="yaml_list", scope="class", params=YAML_LIST)
def fixture_yaml_list(request):
    return scopesim.commands.user_commands.load_yaml_dicts(request.param)

class TestYAML:
    '''Test that yaml files result in correct lists'''

    def test_yaml_read_okay(self, yaml_list):
        '''yaml file is read correctly and gives list'''
        assert isinstance(yaml_list, list)

    def test_yaml_length_not_zero(self, yaml_list):
        '''yaml_list has entries'''
        assert len(yaml_list) > 0

    def test_yaml_entries_are_dicts(self, yaml_list):
        '''yaml list entries are dictionaries'''
        for yaml_entry in yaml_list:
            assert isinstance(yaml_entry, dict)


FILTER_LIST = glob(os.path.join(PKGS_DIR, "METIS/filters/TC_filter_*.dat"))
@pytest.fixture(name="filter_ter", scope="class", params=FILTER_LIST)
def fixture_filter_ter(request):
    return scopesim.effects.FilterCurve(filename=request.param)

class TestFilters:
    '''Test that filter files result in correct FilterCurves'''

    def test_filters_read_okay(self, filter_ter):
        '''filter file is read correctly and gives FilterCurve'''
        assert isinstance(filter_ter, scopesim.effects.FilterCurve)

    def test_filters_table_not_zero(self, filter_ter):
        '''Table attribute shall not be empty'''
        assert len(filter_ter.table) > 0

    def test_filters_has_wavelength_unit(self, filter_ter):
        '''filter file specifies a wavelength unit'''
        assert 'wavelength_unit' in filter_ter.meta

    def test_filter_wavelength_parses_correctly(self, filter_ter):
        '''wavelength unit is parsed by astropy.units'''
        wunit = filter_ter.meta['wavelength_unit']
        assert isinstance(u.Unit(wunit), u.Unit)


QE_LIST = glob(os.path.join(PKGS_DIR, "METIS/QE_detector_*.dat"))
@pytest.fixture(name="qe_curve", scope="class", params=QE_LIST)
def fixture_qe_curve(request):
    return scopesim.effects.QuantumEfficiencyCurve(filename=request.param)

class TestQuantumEfficiency:
    '''Test that QE files result in correct QuantumEfficiencyCurves'''

    def test_qe_read_okay(self, qe_curve):
        '''qe file is read correctly and gives QuantumEfficiencyCurve'''
        assert isinstance(qe_curve, scopesim.effects.QuantumEfficiencyCurve)

    def test_qe_table_not_zero(self, qe_curve):
        '''Table attribute shall not be empty'''
        assert len(qe_curve.table) > 0

    def test_qe_has_wavelength_unit(self, qe_curve):
        '''qe file specifies a wavelength unit'''
        assert 'wavelength_unit' in qe_curve.meta

    def test_qe_wavelength_parses_correctly(self, qe_curve):
        '''wavelength unit is parsed by astropy.units'''
        wunit = qe_curve.meta['wavelength_unit']
        assert isinstance(u.Unit(wunit), u.Unit)


TER_LIST = glob(os.path.join(PKGS_DIR, "METIS/TER_*.dat"))
@pytest.fixture(name="ter_curve", scope="class", params=TER_LIST)
def fixture_ter_curve(request):
    return scopesim.effects.TERCurve(filename=request.param)

class TestTERCurve:
    '''Test that TER files result in correct TERCurves'''

    def test_ter_read_okay(self, ter_curve):
        '''ter file is read correctly and gives TERCurve'''
        assert isinstance(ter_curve, scopesim.effects.TERCurve)

    def test_ter_table_not_zero(self, ter_curve):
        '''Table attribute shall not be empty'''
        assert len(ter_curve.table) > 0

    def test_ter_has_wavelength_unit(self, ter_curve):
        '''ter file specifies a wavelength unit'''
        assert 'wavelength_unit' in ter_curve.meta

    def test_ter_wavelength_parses_correctly(self, ter_curve):
        '''wavelength unit is parsed by astropy.units'''
        wunit = ter_curve.meta['wavelength_unit']
        assert isinstance(u.Unit(wunit), u.Unit)


FPA_LIST = glob(os.path.join(PKGS_DIR, "METIS/FPA_*_layout.dat"))
@pytest.fixture(name="det_list", scope="class", params=FPA_LIST)
def fixture_det_list(request):
    return scopesim.effects.DetectorList(filename=request.param)

class TestFPALayout:
    '''Test that FPA files result in correct DetectorLists'''
    # Do we need explicit tests for units?
    def test_fpa_read_okay(self, det_list):
        '''fpa layout is read correctly and gives DetectorList'''
        assert isinstance(det_list, scopesim.effects.DetectorList)

    def test_fpa_table_not_zero(self, det_list):
        '''Table attribute shall not be empty'''
        assert len(det_list.table) > 0


#### linearity files are currently empty, hence tests xfail
LIN_LIST = glob(os.path.join(PKGS_DIR, "METIS/FPA_linearity_*.dat"))
@pytest.fixture(name="lin_curve", scope="class", params=LIN_LIST)
def fixture_lin_curve(request):
    return scopesim.effects.LinearityCurve(filename=request.param)

class TestLinearityCurve:
    '''Test that linearity files result in correct LinearityCurves'''

    def test_lin_read_okay(self, lin_curve):
        '''linearity curve is read correctly and gives LinearityCurve'''
        assert isinstance(lin_curve, scopesim.effects.LinearityCurve)

    def test_lin_table_not_zero(self, lin_curve):
        '''Table attribute shall not be empty'''
        assert len(lin_curve.table) > 0


MASK_LIST = glob(os.path.join(PKGS_DIR, "METIS/MASK_slit_*.dat"))
@pytest.fixture(name="slit_mask", scope="class", params=MASK_LIST)
def fixture_slit_mask(request):
    return scopesim.effects.ApertureMask(filename=request.param)

class TestSlitMask:
    '''Test that mask files result in correct ApertureMasks'''

    def test_mask_read_okay(self, slit_mask):
        '''Mask file is read correctly and gives ApertureMask'''
        assert isinstance(slit_mask, scopesim.effects.ApertureMask)

    def test_mask_table_not_zero(self, slit_mask):
        '''Table attribute shall not be empty'''
        assert len(slit_mask.table) > 0

    @pytest.mark.parametrize("theunit", ["x_unit", "y_unit"])
    def test_mask_units_parsed_correctly(self, slit_mask, theunit):
        '''units are defined and parsed by astropy'''
        assert isinstance(u.Unit(slit_mask.meta[theunit]), u.Unit)


TRACE_LIST = glob(os.path.join(PKGS_DIR, "METIS/TRACE*.fits"))
@pytest.fixture(name="trace_list", scope="class", params=TRACE_LIST)
def fixture_trace_list(request):
    return scopesim.effects.SpectralTraceList(filename=request.param)

class TestTraceFile:
    '''Test that trace files result in correct SpectralTraces'''

    def test_tracelist_read_okay(self, trace_list):
        '''Trace file is read correctly and gives SpectralTrace'''
        assert isinstance(trace_list, scopesim.effects.SpectralTraceList)

    def test_tracelist_has_table(self, trace_list):
        '''Trace list has a table with at least one entry'''
        assert len(trace_list.data) > 0

    def test_tracelist_has_traces(self, trace_list):
        '''SpectralTraceList contains at least one SpectralTrace'''
        for trace in trace_list.spectral_traces:
            assert isinstance(trace, scopesim.effects.SpectralTrace)

class TestObserves:
    '''Test basic observations for the main instrument modes'''
    def test_something_comes_out_img_lm(self):
        '''Basic test for LM imaging'''
        src = star_field(100, 10, 20, width=10, use_grid=True)

        cmds = scopesim.UserCommands(use_instrument="METIS",
                                     set_modes=['img_lm'])
        metis = scopesim.OpticalTrain(cmds)
        metis['scope_vibration'].include = False
        metis['detector_linearity'].include = False

        metis.observe(src)
        hdus = metis.readout()

        if PLOTS:
            img = hdus[0][1].data
            plt.imshow(img,
                       norm=LogNorm(vmin=0.7*np.median(img),
                                    vmax=1.3*np.median(img)))
            plt.title("LM Imaging Test")
            plt.colorbar()

            plt.show()


    def test_something_comes_out_img_n(self):
        '''Basic test for N imaging'''
        src = star_field(100, 0, 10, width=10, use_grid=True)

        cmds = scopesim.UserCommands(use_instrument="METIS",
                                     set_modes=["img_n"])

        metis = scopesim.OpticalTrain(cmds)
        #metis.cmds.set_modes("img_n")
        metis['scope_vibration'].include = False
        metis['detector_linearity'].include = False

        metis.observe(src)
        hdus = metis.readout()

        if PLOTS:
            img = hdus[0][1].data
            plt.imshow(img, norm=LogNorm(vmin=0.7*np.median(img),
                                         vmax=1.3*np.median(img)))
            plt.title("N Imaging Test")
            plt.colorbar()

            plt.show()
