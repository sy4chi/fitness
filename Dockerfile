FROM eclipse-temurin:21-jdk

WORKDIR /app

COPY . .

RUN javac WorkoutServer.java

EXPOSE 8080

CMD ["java", "WorkoutServer"]
