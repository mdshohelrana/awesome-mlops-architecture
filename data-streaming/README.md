## Building Real-Time Data Streaming using Kafka, Apache Flink and Postgres

# Run the following command to create container
```
cd kafka-producer
ls -la
chmod 777 *
docker build -t kafka-producer .
```

# Run the following command to create container
```
cd flink-processor
mvn clean package
docker build -t flink-processor .
```

# Run the following command to create container
```
cd ..
docker-compose up --build
docker logs --tail 3000 "Container_ID"
```

