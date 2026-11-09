import unittest
from types import SimpleNamespace as NS
from hack_brain.sensors import Measurements
from hack_brain.physical import PhysicalVerifier, in_tray
from hack_brain.context import Context, State, Object, Operation, Hand


def msg(t, positions=None, efforts=None):
    return NS(header=NS(stamp=NS(sec=int(t),nanosec=round((t-int(t))*1e9))),
              name=[str(i) for i in range(1,7)], position=positions or [1.]*6,
              effort=efforts if efforts is not None else [0.]*6)


def fill(m, key, start=1., positions=None, efforts=None):
    for i in range(5):
        m.update(key,msg(start+i*.1,positions,efforts))


class MeasurementsTests(unittest.TestCase):
    def test_requires_new_continuous_finite_frames_after_terminal(self):
        m=Measurements()
        fill(m,'left')
        self.assertIsNotNone(m.window('left',1.4,1))
        self.assertIsNone(m.window('left',1.4,1.3))
        self.assertIsNone(m.window('left',2,1))
        # Repeated frames cannot establish a settling interval.
        m=Measurements()
        for _ in range(20): m.update('left',msg(1))
        self.assertIsNone(m.window('left',1,1))
        m.update('left',msg(1.1,[float('nan')]*6))
        self.assertEqual(len(m.samples['left']),0)

    def test_squeezing_thumb_jitter_still_reads_holding(self):
        m=Measurements()
        for i in range(5):
            m.update('left',msg(1+i*.1,[.4,.4,.4,.4,.7+(.02 if i%2 else 0),.2],[.3]*6))
        self.assertEqual(m.hand('left',1.4,1,[.2,.2,.2,.2,.5,.2])[1],'HOLDING')

    def test_finger_curled_below_closed_still_reads(self):
        m=Measurements();fill(m,'right',positions=[1.01,.94,-.13,1.12,.55,.1],efforts=[.01,.12,1.,1.,.51,1.])
        self.assertEqual(m.hand('right',1.4,1,[1,1,1,.3,.4,.1])[1],'HOLDING')

    def test_pinching_thumb_oscillation_still_reads(self):
        m=Measurements()
        for i in range(5):
            m.update('left',msg(1+i*.1,[1.03,1.03,1.04,.88,.36+(.08 if i%2 else 0),.11],[.66,.65,.63,1.,1.,1.]))
        self.assertIsNotNone(m.hand('left',1.4,1,[1,1,1,.3,.4,.1]))

    def test_closing_finger_is_not_at_rest(self):
        m=Measurements()
        for i in range(5):
            m.update('left',msg(1+i*.1,[1.,1.,1.,1.-i*.1,1.-i*.1,.1],[.1]*6))
        self.assertIsNone(m.hand('left',1.4,1,[1,1,1,.3,.4,.1]))

    def test_finger_pushed_far_past_open_still_reads(self):
        m=Measurements();fill(m,'right',positions=[.2,2.19,2.2,.19,.52,.2],efforts=[.17,1.,1.,.31,.08,.04])
        self.assertIsNotNone(m.hand('right',1.4,1,[.2,.2,.2,.2,.5,.2]))

    def test_finger_pushed_past_open_still_reads(self):
        m=Measurements();fill(m,'left',positions=[1.02,1.02,1.01,.83,1.54,.11],efforts=[.67,.66,.68,1.,.61,.29])
        self.assertIsNotNone(m.hand('left',1.4,1,[1,1,1,.3,1,.1]))

    def test_open_right_hand_reads_empty(self):
        m=Measurements();fill(m,'right',positions=[.99,.99,.99,1.,.94,1.],efforts=[.01,.02,.02,.03,.02,0.])
        self.assertEqual(m.hand('right',1.4,1)[1],'EMPTY')

    def test_pinch_ready_reads_empty_without_a_target(self):
        m=Measurements();fill(m,'left',positions=[.99,.99,.99,.99,.97,.1])
        self.assertEqual(m.hand('left',1.4,1)[1],'EMPTY')

    def test_still_report_names_the_moving_joint(self):
        m=Measurements()
        for i in range(5): m.update('joints',msg(1+i*.1,[1.,1.,1.+i*.01,1.,1.,1.]))
        report=m.still_report('joints',1.4,1)
        self.assertIn('3 0.0400',report)
        self.assertNotIn('1 0.',report)
        self.assertIn('no samples',Measurements().still_report('joints',1.4,1))

    def test_motion_missing_names_and_large_gaps_do_not_prove_stop(self):
        m=Measurements()
        for i in range(5): m.update('left',msg(1+i*.1,[i*.1]*6))
        self.assertIsNone(m.window('left',1.4,1))
        m=Measurements()
        for t in (1,1.3,1.4): m.update('left',msg(t))
        self.assertIsNone(m.window('left',1.4,1))
        fill(m,'right')
        self.assertIsNone(m.window('right',1.4,1,['missing']))

    def test_empty_vs_opposed_contact_and_missing_effort(self):
        target=[.2,.2,.2,.2,.5,.2]
        m=Measurements();fill(m,'left')
        self.assertEqual(m.hand('left',1.4,1)[1],'EMPTY')
        m=Measurements();fill(m,'left',positions=target)
        self.assertEqual(m.hand('left',1.4,1,target)[1],'EMPTY')
        blocked=[.4,.4,.4,.4,.7,.2]
        m=Measurements();fill(m,'left',positions=blocked,efforts=[])
        self.assertIsNone(m.hand('left',1.4,1,target))
        m=Measurements();fill(m,'left',positions=blocked,efforts=[.3]*6)
        self.assertEqual(m.hand('left',1.4,1,target)[1],'HOLDING')
        m=Measurements();fill(m,'left',positions=blocked,efforts=[0.]*6)
        self.assertIsNone(m.hand('left',1.4,1,target))  # stopped short without pushing proves nothing

    def test_bench_pinch_reads_holding(self):
        pinch=[1.,1.,1.,.3,.4,.1]
        m=Measurements();fill(m,'right',positions=[1.,1.,1.,1.03,.25,.1],efforts=[0,0,0,.4,.3,0])
        self.assertEqual(m.hand('right',1.4,1,pinch)[1],'HOLDING')
        # The thumb overshoots its target on an empty close too: that is still an empty hand.
        m=Measurements();fill(m,'right',positions=[1.,1.,1.,.3,.25,.1],efforts=[0,0,0,.05,.05,0])
        self.assertEqual(m.hand('right',1.4,1,pinch)[1],'EMPTY')
        m=Measurements();fill(m,'right',positions=[1.,1.,1.,2.,.25,.1])
        self.assertIsNone(m.hand('right',1.4,1,pinch))


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.m=Measurements();self.height=0
        self.v=PhysicalVerifier(self.m, lambda side,now:(now,(.4,0,self.height)))
        self.c=Context(state=State.VERIFY_GRASP,sim=1.4,target=Object('a','figure',name='a'),
                       active=Operation(1,State.VERIFY_GRASP,1,0),destination='tray')
        self.target=[.2,.2,.2,.2,.5,.2]
        self.v.closing(self.c.target,'left',self.target,1)

    def contact(self,start=1):
        fill(self.m,'left',start,[.4,.4,.4,.4,.7,.2],[.3]*6)

    def test_measured_partial_failed_lift_can_recover(self):
        self.contact();ev=self.v.read(self.c)
        self.assertEqual(ev.hand,Hand.HOLDING);self.assertFalse(ev.lifted)
        self.height=.08;self.c.state=State.RECOVER
        ev=self.v.read(self.c)
        self.assertTrue(ev.lifted)

    def test_missing_baseline_is_taken_in_verify_grasp(self):
        v=PhysicalVerifier(self.m, lambda side,now:None if now < 1.2 else (now,(.4,0,self.height)))
        v.closing(self.c.target,'left',self.target,1)
        self.assertIsNone(v.baseline)
        self.contact();ev=v.read(self.c)
        self.assertEqual(ev.hand,Hand.HOLDING);self.assertIsNotNone(v.baseline)
        self.height=.15;self.c.state=State.VERIFY_LIFT
        self.assertTrue(v.read(self.c).lifted)

    def test_verify_lift_waits_for_a_pose_after_the_lift(self):
        self.contact();self.v.read(self.c)
        self.height=.15;self.c.state=State.VERIFY_LIFT
        self.c.active=Operation(2,State.VERIFY_LIFT,1.5,0)
        self.assertIsNone(self.v.read(self.c))  # TCP stamped 1.4, before the phase start 1.5
        self.c.sim=2.0;fill(self.m,'left',1.5,[.4,.4,.4,.4,.7,.2],[.3]*6)
        self.assertTrue(self.v.read(self.c).lifted)

    def test_tcp_motion_alone_or_stale_hand_cannot_prove_lift(self):
        self.height=.2
        self.assertIsNone(self.v.read(self.c))
        fill(self.m,'left',positions=self.target)
        self.assertEqual(self.v.read(self.c).hand,Hand.EMPTY)
        self.c.sim=3
        self.assertIsNone(self.v.read(self.c))

    def test_contact_loss_cannot_reassociate_object(self):
        self.contact();self.v.read(self.c)
        self.v.contact_lost=True;self.height=.2
        self.assertIsNone(self.v.read(self.c))

    def scene(self,name='a',x=0.):
        return NS(header=NS(stamp=NS(sec=1,nanosec=400000000),frame_id='pelvis'),
                  objects=[NS(name=name,confidence=.9,partially_visible=False,yaw=0.,
                              center=NS(x=x,y=0.,z=.03),size=NS(x=.04,y=.04,z=.06))],
                  containers=[NS(id='tray',kind='tray',pose=NS(position=NS(x=0.,y=0.,z=.07),
                                orientation=NS(x=0.,y=0.,z=0.,w=1.)),size=NS(x=.22,y=.32,z=.07))])

    def test_placement_needs_unique_named_detection_in_observed_tray(self):
        fill(self.m,'left');self.c.state=State.VERIFY_PLACE
        self.assertFalse(self.v.read(self.c).placed)
        self.assertTrue(self.v.read(self.c,self.scene()).placed)
        for scene in (self.scene('other'),self.scene(x=.2)):
            self.assertFalse(self.v.read(self.c,scene).placed)
        scene=self.scene();scene.objects.append(scene.objects[0])
        self.assertFalse(self.v.read(self.c,scene).placed)
        scene=self.scene();scene.header.stamp.sec=0
        self.assertIsNone(self.v.read(self.c,scene))
        self.c.target=Object('track','figure')
        self.assertFalse(self.v.read(self.c,self.scene()).placed)

    def test_open_both_hands_for_initial_verification(self):
        self.c.state=State.VERIFY_EMPTY
        fill(self.m,'left');self.assertIsNone(self.v.read(self.c))
        fill(self.m,'right');self.assertEqual(self.v.read(self.c).hand,Hand.EMPTY)

    def test_hand_closed_on_nothing_reads_empty_after_the_other_arm_placed(self):
        # The left pinch slipped and stayed shut; the next attempt went with the right arm (OPEN clears closure).
        self.v.closure=None;self.v.arm='right'
        self.c.state=State.VERIFY_EMPTY
        fill(self.m,'left',1,self.target)
        fill(self.m,'right')
        self.assertEqual(self.v.read(self.c).hand,Hand.EMPTY)

    def test_hand_shut_on_an_object_is_not_empty(self):
        self.c.state=State.VERIFY_EMPTY
        self.contact();fill(self.m,'right')
        self.assertIsNone(self.v.read(self.c))


if __name__=='__main__': unittest.main()
