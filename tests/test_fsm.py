import unittest
from dataclasses import replace
from hack_brain.context import Context, Config, Event, Evidence, Hand, Lift, Object, Scene, State
from hack_brain.fsm import transition
from hack_brain.policies import destinations
from hack_brain.task import parse_task
from fakes import Fake, sort_spec, tubes_spec


class MissionTests(unittest.TestCase):
    def test_multiple_objects_and_task_mapping(self):
        f = Fake()
        c = f.run()
        self.assertEqual(c.reason, 'candidates_exhausted')
        self.assertTrue(all(h.placed and h.lift == Lift.LIFTED for h in c.history.values()))
        self.assertEqual([(x.target.id, x.destination) for x in f.commands if x.phase == State.TRANSFER],
                         [('a', 'blue'), ('b', 'red')])
        self.assertEqual(sum(e.kind == 'DONE' for e in f.effects), 1)

    def test_no_candidates(self):
        f = Fake(objects=[])
        self.assertEqual(f.run().reason, 'candidates_exhausted')
        self.assertFalse(any(e.phase == State.OPEN for e in f.commands))

    def test_unreachable_retries_are_bounded(self):
        f = Fake(objects=[Object('a', 'figure', name='a')])
        for _ in range(2):
            f.until(State.APPROACH)
            f.answer('unreachable', ok=False)
        c = f.run()
        self.assertEqual(c.history['a'].attempts, 2)
        self.assertFalse(any(e.phase == State.CLOSE for e in f.commands))

    def test_empty_grasp_does_not_lift(self):
        f = Fake()
        f.until(State.VERIFY_GRASP)
        f.answer(Evidence(f.sim, Hand.EMPTY, source='fake'))
        self.assertEqual(f.c.state, State.RECOVER)
        self.assertFalse(any(e.phase == State.LIFT for e in f.commands))

    def test_unknown_does_not_transfer(self):
        for state in (State.VERIFY_EMPTY, State.VERIFY_GRASP, State.VERIFY_LIFT, State.VERIFY_HOLD):
            with self.subTest(state=state):
                f = Fake()
                f.until(state)
                n = len(f.commands)
                f.answer(Evidence(f.sim, source='fake'))
                self.assertEqual(f.c.state, State.DONE)
                self.assertEqual(len(f.commands), n)
                self.assertEqual(f.c.hand, Hand.UNKNOWN)

    def test_loss_after_lift_keeps_history_and_reobserves(self):
        f = Fake()
        f.until(State.VERIFY_HOLD)
        f.answer(Evidence(f.sim, Hand.EMPTY, source='fake'))
        self.assertEqual(f.c.history['a'].lift, Lift.LIFTED)
        f.answer(Evidence(f.sim, Hand.EMPTY, source='fake'))
        self.assertEqual(f.c.state, State.OBSERVE)
        f.answer()
        self.assertEqual(f.c.target.id, 'a')

    def test_transfer_failure_holding_replans_without_release(self):
        f = Fake()
        f.until(State.TRANSFER)
        f.answer('plan_failed', ok=False)
        self.assertEqual(f.c.state, State.RECOVER)
        f.answer(Evidence(f.sim, Hand.HOLDING, 'a', source='fake'))
        self.assertEqual(f.c.state, State.DESTINATION)
        self.assertFalse(any(e.phase == State.RELEASE for e in f.commands))

    def test_motion_error_without_stop_does_not_recover(self):
        f = Fake()
        f.until(State.TRANSFER)
        f.answer('timeout', ok=False, stopped=False)
        self.assertEqual(f.c.state, State.CANCELLING)
        self.assertEqual(f.effects[-1].kind, 'CANCEL')

    def test_release_failure_not_placement(self):
        f = Fake()
        f.until(State.RELEASE)
        f.answer('hand_failed', ok=False)
        self.assertFalse(f.c.history['a'].placed)
        self.assertEqual(f.c.state, State.RECOVER)

    def test_unconfirmed_placement(self):
        f = Fake()
        f.until(State.VERIFY_PLACE)
        f.answer(replace(f.default(), placed=False))
        self.assertEqual(f.c.reason, 'placement_unconfirmed')
        self.assertFalse(f.c.history['a'].placed)

    def test_wrong_destination_not_success(self):
        f = Fake()
        f.until(State.VERIFY_PLACE)
        f.answer(replace(f.default(), destination='red'))
        self.assertFalse(f.c.history['a'].placed)

    def test_timeout_cancel_success(self):
        f = Fake()
        f.until(State.APPROACH)
        op_id = f.c.active.id
        f.tick(sim=22, wall=101)
        self.assertEqual(f.c.state, State.CANCELLING)
        f.send(Event('CANCEL_ACK', op_id))
        self.assertEqual(f.c.state, State.CANCELLING)
        f.send(Event('RESULT', op_id, ok=False, stopped=True))
        self.assertEqual(f.c.state, State.DONE)
        self.assertFalse(f.c.blocked)
        self.assertIsNone(f.c.active)

    def test_cancel_result_before_ack(self):
        f = Fake()
        op = f.c.active.id
        f.send(Event('CANCEL'))
        f.send(Event('RESULT', op, stopped=True))
        self.assertEqual(f.c.state, State.CANCELLING)
        f.send(Event('CANCEL_ACK', op))
        self.assertFalse(f.c.blocked)
        self.assertEqual(f.c.state, State.DONE)

    def test_unconfirmed_cancellation(self):
        for mode in ('no_ack', 'refused', 'no_stop'):
            with self.subTest(mode=mode):
                f = Fake()
                op = f.c.active.id
                f.send(Event('CANCEL'))
                if mode == 'refused':
                    f.send(Event('CANCEL_ACK', op, ok=False))
                elif mode == 'no_stop':
                    f.send(Event('CANCEL_ACK', op))
                    f.send(Event('RESULT', op, stopped=False))
                else:
                    f.tick(wall=106)
                self.assertEqual(f.c.state, State.DONE)
                self.assertTrue(f.c.blocked)
                self.assertIsNotNone(f.c.active)

    def test_old_duplicate_and_acceptance(self):
        f = Fake()
        old = f.c.active.id
        f.send(Event('ACCEPTED', old))
        self.assertEqual(f.c.state, State.VERIFY_EMPTY)
        f.answer()
        current = f.c.active.id
        n = len(f.commands)
        f.send(Event('RESULT', old, Evidence(1, Hand.EMPTY, source='fake')))
        f.send(Event('ACCEPTED', old))
        self.assertEqual(f.c.active.id, current)
        self.assertEqual(len(f.commands), n)

    def test_global_retry_limit(self):
        f = Fake(config=replace(Config(), max_recoveries=2))
        for _ in range(3):
            f.until(State.OBSERVE)
            f.answer('dependency_failed', ok=False)
        self.assertTrue(f.c.reason.startswith('retries_exhausted'))

    def test_budget_before_grasp(self):
        f = Fake()
        f.until(State.OBSERVE)
        f.tick(sim=230, wall=101)
        # Large jump also expires the read operation: cancellation is required.
        self.assertFalse(any(e.phase == State.OPEN for e in f.commands))
        g = Fake(config=replace(Config(), cycle_estimate=300))
        self.assertEqual(g.run().reason, 'insufficient_time')

    def test_budget_holding_prioritizes_delivery(self):
        f = Fake(config=replace(Config(), operation_timeout=300))
        f.until(State.VERIFY_LIFT)
        f.tick(sim=240, wall=101)
        f.answer()
        self.assertEqual(f.c.state, State.DESTINATION)
        f.answer()
        self.assertEqual(f.c.state, State.VERIFY_TRANSFER)
        f.answer()
        self.assertEqual(f.c.state, State.TRANSFER)
        f.tick(sim=266, wall=102)
        self.assertEqual(f.c.state, State.CANCELLING)
        self.assertEqual(f.c.reason, 'deadline')

    def test_clock_stall_reset_missing(self):
        f = Fake()
        f.tick(wall=116)
        self.assertEqual(f.c.reason, 'clock_stalled')
        g = Fake()
        g.tick(sim=0)
        self.assertEqual(g.c.reason, 'clock_reset')
        c, _ = transition(Context(), Event('TICK', data=(None, 100)))
        c, effects = transition(c, Event('TICK', data=(None, 116)))
        self.assertEqual(c.reason, 'clock_missing')
        self.assertEqual(effects[-1].kind, 'DONE')

    def test_late_start_does_not_extend_budget(self):
        c, _ = transition(Context(), Event('START', data=(sort_spec(), frozenset({
            'perception', 'pick', 'verification', 'transfer', 'release', 'retreat', 'place'}))))
        c, _ = transition(c, Event('TICK', data=(999, 100)))
        self.assertEqual(c.reason, 'budget_exhausted')
        self.assertEqual(c.deadline, 270)

    def test_send_exception_is_not_a_rejection(self):
        f = Fake()
        f.send(Event('SEND_ERROR', f.c.active.id, 'network'))
        self.assertEqual(f.c.state, State.CANCELLING)
        f.tick(wall=106)
        self.assertTrue(f.c.blocked)

    def test_exception_finishes_after_bounded_cancel(self):
        f = Fake()
        f.send(Event('ERROR', data='exception'))
        f.tick(wall=106)
        self.assertEqual(f.c.state, State.DONE)
        self.assertTrue(any(e.kind == 'DONE' for e in f.effects))

    def test_all_tubes_variants(self):
        for variant in ('easy', 'standard', 'color', 'lying'):
            with self.subTest(variant=variant):
                f = Fake(tubes_spec(variant), [Object('tube', 'falcon50', name='tube', cap='red', lying=variant == 'lying')])
                self.assertTrue(f.run().history['tube'].placed)
                phases = [e.phase for e in f.commands]
                self.assertIn(State.INSERT, phases)
                self.assertEqual(State.REORIENT in phases, variant == 'lying')
                self.assertEqual(next(e.destination for e in f.commands if e.phase == State.TRANSFER), 'hood50')

    def test_tube_rules_and_table_exclusion(self):
        obj = Object('x', 'falcon15', cap='red')
        self.assertEqual(destinations(tubes_spec('standard'), obj), ['hood15'])
        self.assertEqual(destinations(tubes_spec('color'), obj), ['hood50'])
        self.assertEqual(destinations(tubes_spec('easy'), obj), ['hood50', 'hood15'])

    def test_tube_slots_not_reused(self):
        f = Fake(tubes_spec('easy'), [Object(x, 'falcon50', name=x) for x in ('a', 'b')])
        f.run()
        slots = [e.slot for e in f.commands if e.phase == State.INSERT]
        self.assertEqual(slots, [0, 1])

    def test_unavailable_skills_done_without_motion(self):
        for spec in (sort_spec(), *(tubes_spec(v) for v in ('easy', 'standard', 'color', 'lying'))):
            f = Fake(spec, capabilities=frozenset())
            self.assertEqual(f.c.state, State.DONE)
            self.assertTrue(f.c.reason.startswith('unavailable:'))
            self.assertEqual(f.commands, [])

    def test_terminal_absorbs_everything(self):
        f = Fake(objects=[])
        f.run()
        original = f.c
        for event in (Event('START'), Event('RESULT', 5), Event('ERROR'), Event('TICK', data=(0, 999))):
            c, effects = transition(original, event)
            self.assertEqual(c, original)
            self.assertEqual(effects, [])

    def test_stale_and_wrong_object_evidence(self):
        for ev in (Evidence(-5, Hand.HOLDING, 'a', source='fake'),
                   Evidence(1, Hand.HOLDING, 'wrong', source='fake')):
            f = Fake()
            f.until(State.VERIFY_GRASP)
            f.answer(ev)
            self.assertEqual(f.c.reason, 'physical_unknown')

    def test_contact_invalidates_scene_and_anonymous_identity(self):
        f = Fake(objects=[Object('track', 'figure')])
        f.until(State.DESCEND)
        self.assertIsNone(f.c.scene)
        f.until(State.VERIFY_GRASP)
        f.answer(Evidence(1, Hand.EMPTY, source='fake'))
        f.answer(Evidence(1, Hand.EMPTY, source='fake'))
        self.assertEqual(f.c.reason, 'identity_lost_after_contact')

    def test_stale_scene_bounded(self):
        f = Fake(config=replace(Config(), max_recoveries=1))
        f.until(State.OBSERVE)
        f.answer(Scene(-9))
        f.answer(Scene(-9))
        self.assertEqual(f.c.state, State.DONE)

    def test_retreat_failure_preserves_confirmed_place(self):
        f = Fake()
        f.until(State.RETREAT)
        f.answer('retreat_failed', ok=False)
        self.assertTrue(f.c.history['a'].placed)
        self.assertEqual(f.c.reason, 'retreat_failed')

    def test_unknown_task_and_rule(self):
        f = Fake(parse_task('{"task":"other"}'))
        self.assertEqual(f.c.reason, 'unsupported_task')
        spec = tubes_spec('color')
        spec.rule = 'typo'
        self.assertEqual(spec.hood_racks_for('falcon50', 'red'), [])
        self.assertEqual(Fake(spec).c.reason, 'unsupported_task')

    def test_sensor_loss_during_motion_cancels(self):
        f = Fake()
        f.until(State.TRANSFER)
        f.send(Event('SENSOR', data=Evidence(1, Hand.EMPTY, source='fake')))
        self.assertEqual(f.c.state, State.CANCELLING)
        self.assertEqual(f.c.hand, Hand.EMPTY)
        self.assertEqual(f.effects[-1].kind, 'CANCEL')

    def test_destination_requires_fresh_hold(self):
        f = Fake()
        f.until(State.VERIFY_TRANSFER)
        f.answer(Evidence(1, source='fake'))
        self.assertEqual(f.c.state, State.DONE)
        self.assertFalse(any(x.phase == State.TRANSFER for x in f.commands))

    def test_stale_before_contact_never_descends(self):
        f = Fake()
        f.until(State.APPROACH)
        f.tick(sim=4, wall=101)
        f.answer()
        self.assertEqual(f.c.state, State.RECOVER)
        self.assertFalse(any(x.phase == State.DESCEND for x in f.commands))

    def test_accepted_and_ticks_do_not_resend(self):
        f = Fake()
        f.until(State.APPROACH)
        n = len(f.commands)
        for i in range(10):
            f.send(Event('ACCEPTED', f.c.active.id))
            f.tick(sim=1 + i / 10, wall=100 + i / 10)
        self.assertEqual(len(f.commands), n)

    def test_unknown_event_is_noop(self):
        f = Fake()
        before = f.c
        effects = f.send(Event('unexpected', f.c.active.id))
        self.assertEqual(f.c, before)
        self.assertEqual(effects, [])

    def test_nan_confidence_is_not_candidate(self):
        f = Fake(objects=[Object('x', 'figure', confidence=float('nan'))])
        self.assertEqual(f.run().reason, 'candidates_exhausted')

    def test_reorientation_failure_cannot_bypass_upright_verification(self):
        f = Fake(tubes_spec('lying'), [Object('tube', 'falcon50', name='tube', lying=True)])
        f.until(State.REORIENT)
        f.answer('reorientation_failed', ok=False)
        f.answer(Evidence(f.sim, Hand.HOLDING, 'tube', source='fake'))
        self.assertEqual(f.c.state, State.REORIENT)
        self.assertFalse(any(x.phase == State.INSERT for x in f.commands))

    def test_progressing_clocks_happy_path(self):
        f = Fake()
        for _ in range(150):
            if f.c.state == State.DONE:
                break
            f.tick(sim=f.sim + 0.1, wall=f.wall + 0.2)
            f.answer()
        self.assertEqual(f.c.state, State.DONE)
        self.assertTrue(all(h.placed for h in f.c.history.values()))

    def test_sort3_and_extra_pool_categories(self):
        for task, categories in (('sort3', {'figure': 'blue', 'dish': 'red', 'box': 'blue'}),
                                 ('sort2', {'electronics': 'red', 'box': 'blue'})):
            spec = sort_spec()
            spec.task, spec.categories = task, categories
            spec.objects = {str(i): cat for i, cat in enumerate(list(categories) * 3)}
            f = Fake(spec, [Object(cat, cat, name=cat) for cat in categories])
            f.run()
            self.assertEqual({x.target.category: x.destination for x in f.commands if x.phase == State.TRANSFER},
                             categories)

    def test_lost_connection_does_not_claim_stop(self):
        f = Fake()
        f.send(Event('CONNECTION_LOST'))
        self.assertEqual(f.c.state, State.DONE)
        self.assertTrue(f.c.blocked)
        self.assertEqual(f.effects[-1].kind, 'DONE')

    def test_reducer_does_not_mutate_input(self):
        f = Fake()
        before = f.c
        active = before.active.id
        f.answer()
        self.assertEqual(before.state, State.VERIFY_EMPTY)
        self.assertEqual(before.active.id, active)


if __name__ == '__main__':
    unittest.main()
