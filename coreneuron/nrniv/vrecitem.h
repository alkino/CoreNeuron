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

#ifndef vrecitem_h
#define vrecitem_h

#include "coreneuron/nrniv/netcon.h"
#include "coreneuron/nrniv/ivocvect.h"

class PlayRecord;

#define VecPlayContinuousType 4

// used by PlayRecord subclasses that utilize discrete events
class PlayRecordEvent : public DiscreteEvent {
public:
	PlayRecordEvent();
	virtual ~PlayRecordEvent();
	virtual void deliver(double, NetCvode*, NrnThread*);
	virtual void pr(const char*, double t, NetCvode*);
	virtual void frecord_init(TQItem* q);
	virtual NrnThread* thread();
	PlayRecord* plr_;
	static unsigned long playrecord_send_;
	static unsigned long playrecord_deliver_;
};

// common interface for Play and Record for all integration methods.
class PlayRecord {
public:
	PlayRecord(double* pd, int ith);
	virtual ~PlayRecord();
	virtual void play_init(){}	// called near beginning of finitialize
	virtual void continuous(double){} // play - every f(y, t) or res(y', y, t); record - advance_tn and initialize flag
	virtual void deliver(double, NetCvode*){} // at associated DiscreteEvent
	virtual PlayRecordEvent* event() { return nil;}
	virtual void pr(); // print identifying info
	virtual int type() { return 0; }

	// administration
	virtual void frecord_init(TQItem*) {}

	double* pd_;
	int ith_; // The thread index
};


class VecPlayContinuous : public PlayRecord {
public:
	VecPlayContinuous(double*, IvocVect* yvec, IvocVect* tvec, IvocVect* discon, int ith);
	virtual ~VecPlayContinuous();
	void init(IvocVect* yvec, IvocVect* tvec, IvocVect* tdiscon);
	virtual void play_init();
	virtual void deliver(double tt, NetCvode*);
	virtual PlayRecordEvent* event() { return e_;}
	virtual void pr();

	void continuous(double tt);
	double interpolate(double tt);
	double interp(double th, double x0, double x1){ return x0 + (x1 - x0)*th; }
	void search(double tt);

	virtual int type() { return VecPlayContinuousType; }

	IvocVect* y_;
	IvocVect* t_;
	IvocVect* discon_indices_;
    size_t last_index_;
    size_t discon_index_;
    size_t ubound_index_;

	PlayRecordEvent* e_;
};

#endif
