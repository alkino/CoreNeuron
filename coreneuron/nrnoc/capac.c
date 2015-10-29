/*
Copyright (c) 2014 EPFL-BBP, All rights reserved.

THIS SOFTWARE IS PROVIDED BY THE BLUE BRAIN PROJECT "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE BLUE BRAIN PROJECT
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "coreneuron/nrnconf.h"
#include "coreneuron/nrnoc/multicore.h"
#include "coreneuron/nrnoc/membdef.h"

#if defined(_OPENACC)
#define _PRAGMA_FOR_INIT_ACC_LOOP_ _Pragma("acc parallel loop present(vdata[0:_cntml_padded*nparm]) if(_nt->compute_gpu)")
#define _PRAGMA_FOR_CUR_ACC_LOOP_ _Pragma("acc parallel loop present(vdata[0:_cntml_padded*nparm], ni[0:_cntml_actual], _vec_rhs[0:_nt->end]) if(_nt->compute_gpu) async(stream_id)")
#define _PRAGMA_FOR_JACOB_ACC_LOOP_ _Pragma("acc parallel loop present(vdata[0:_cntml_padded*nparm], ni[0:_cntml_actual], _vec_d[0:_nt->end]) if(_nt->compute_gpu) async(stream_id)")
#else
#define _PRAGMA_FOR_INIT_ACC_LOOP_ _Pragma("")
#define _PRAGMA_FOR_CUR_ACC_LOOP_ _Pragma("")
#define _PRAGMA_FOR_JACOB_ACC_LOOP_ _Pragma("")
#endif

#if !defined(LAYOUT)
/* 1 means AoS, >1 means AoSoA, <= 0 means SOA */
#define LAYOUT 1
#endif
#if LAYOUT >= 1
#define _STRIDE LAYOUT
#else
#define _STRIDE _cntml_padded + _iml
#endif

static const char *mechanism[] = { "0", "capacitance", "cm",0, "i_cap", 0,0 };
static void cap_alloc(double*, Datum*, int type);
static void cap_init(NrnThread*, Memb_list*, int);

#define nparm 2

void capac_reg_(void) {
	int mechtype;
	/* all methods deal with capacitance in special ways */
	register_mech(mechanism, cap_alloc, (mod_f_t)0, (mod_f_t)0, (mod_f_t)0, (mod_f_t)cap_init, -1, 1);
	mechtype = nrn_get_mechtype(mechanism[1]);
	_nrn_layout_reg(mechtype, LAYOUT);
	hoc_register_prop_size(mechtype, nparm, 0);
}

#define cm  vdata[0*_STRIDE]
#define i_cap  vdata[1*_STRIDE]

/*
cj is analogous to 1/dt for cvode and daspk
for fixed step second order it is 2/dt and
for pure implicit fixed step it is 1/dt
It used to be static but is now a thread data variable
*/

void nrn_cap_jacob(NrnThread* _nt, Memb_list* ml) {
	int _cntml_actual = ml->nodecount;
	int _cntml_padded = ml->_nodecount_padded;
	int _iml;
	double *vdata = ml->data;
	double cfac = .001 * _nt->cj;
    (void) _cntml_padded; /* unused when layout=1*/

    double* _vec_d = _nt->_actual_d;
    int stream_id = _nt->stream_id;

	{ /*if (use_cachevec) {*/
		int* ni = ml->nodeindices;

        _PRAGMA_FOR_JACOB_ACC_LOOP_
		for (_iml=0; _iml < _cntml_actual; _iml++) {
			_vec_d[ni[_iml]] += cfac*cm;
		}
	}
}

static void cap_init(NrnThread* _nt, Memb_list* ml, int type ) {
	int _cntml_actual = ml->nodecount;
	int _cntml_padded = ml->_nodecount_padded;
	int _iml;
	double *vdata = ml->data;
	
    (void)_nt; (void)type; (void) _cntml_padded; /* unused */

    _PRAGMA_FOR_INIT_ACC_LOOP_
	for (_iml=0; _iml < _cntml_actual; ++_iml) {
		i_cap = 0;
	}
}

void nrn_capacity_current(NrnThread* _nt, Memb_list* ml) {
	int _cntml_actual = ml->nodecount;
	int _cntml_padded = ml->_nodecount_padded;
	int _iml;
	double *vdata = ml->data;
	double cfac = .001 * _nt->cj;
    
    /*@todo: verify cfac is being copied !! */

    (void) _cntml_padded; /* unused when layout=1*/

	/* since rhs is dvm for a full or half implicit step */
    /* (nrn_update_2d() replaces dvi by dvi-dvx) */
    /* no need to distinguish secondorder */
    int* ni = ml->nodeindices;
    double* _vec_rhs = _nt->_actual_rhs;
    int stream_id = _nt->stream_id;

    _PRAGMA_FOR_CUR_ACC_LOOP_
    for (_iml=0; _iml < _cntml_actual; _iml++) {
        i_cap = cfac*cm*_vec_rhs[ni[_iml]];
    }
}

/* the rest can be constructed automatically from the above info*/

static void cap_alloc(double* data, Datum* pdata, int type) {
	(void)pdata; (void)type; /* unused */
	data[0] = DEF_cm;	/*default capacitance/cm^2*/
}
