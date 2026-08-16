from nlp_pipeline.uuid_gen import UUIDGenerator, default_uuid_generator


def test_same_seed_produces_same_sequence():
    gen_a = UUIDGenerator(random_state=1)
    gen_b = UUIDGenerator(random_state=1)

    sequence_a = [gen_a.next() for _ in range(5)]
    sequence_b = [gen_b.next() for _ in range(5)]

    assert sequence_a == sequence_b


def test_different_seeds_produce_different_sequences():
    gen_a = UUIDGenerator(random_state=1)
    gen_b = UUIDGenerator(random_state=2)

    assert gen_a.next() != gen_b.next()


def test_reset_without_args_replays_same_state():
    gen = UUIDGenerator(random_state=7)
    first_pass = [gen.next() for _ in range(3)]

    gen.reset()
    second_pass = [gen.next() for _ in range(3)]

    assert first_pass == second_pass


def test_reset_with_new_random_state_changes_output():
    gen = UUIDGenerator(random_state=7)
    baseline = gen.next()

    gen.reset(random_state=8)
    assert gen.next() != baseline

    # the new random_state should now be "sticky" across a plain reset()
    gen.reset(random_state=8)
    replay = [gen.next() for _ in range(3)]
    gen.reset()
    assert replay == [gen.next() for _ in range(3)]


def test_next_returns_valid_hex_uuid4_string():
    gen = UUIDGenerator(random_state=0)
    value = gen.next()

    assert isinstance(value, str)
    assert len(value) == 32
    # round-trips through uuid.UUID and is tagged as version 4
    import uuid

    assert uuid.UUID(hex=value).version == 4


def test_default_uuid_generator_is_a_singleton():
    assert default_uuid_generator() is default_uuid_generator()
