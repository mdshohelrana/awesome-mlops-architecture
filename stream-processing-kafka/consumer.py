from kafka import KafkaConsumer

# Initialize Kafka consumer
consumer = KafkaConsumer(
    'test',  # Make sure this is the same topic used by the producer
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',  # Start from the earliest messages
    enable_auto_commit=True,
    group_id='my-group'  # Consumer group ID
)

# Consume messages from Kafka topic 'test'
for message in consumer:
    print(f"Received: {message.value.decode('utf-8')}")
