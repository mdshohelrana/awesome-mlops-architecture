from kafka import KafkaProducer
import time

# Initialize Kafka producer
producer = KafkaProducer(bootstrap_servers='localhost:9092')

# Send messages to Kafka topic 'test'
for i in range(100000):
    message = f"Message {i}"
    producer.send('test', message.encode('utf-8'))
    print(f"Sent: {message}")
    time.sleep(1)  # Simulate some delay between messages

# Flush and close the producer
producer.flush()
producer.close()
