# Premium Edge AI PT Studio - Java Web Server

Java `HttpServer` 기반의 웹 운동 분석 서비스입니다.
프론트엔드 파일과 로그인/운동 기록 API를 모두 `WorkoutServer.java`가 제공합니다.

## 실행 방법

Java 실행이 가능한 배포 서버에서 저장소를 받은 뒤 실행합니다.

```bash
javac WorkoutServer.java
java WorkoutServer
```

배포 환경에서 포트를 지정해야 한다면 `PORT` 환경 변수를 사용할 수 있습니다.

```bash
PORT=8080 java WorkoutServer
```

실행 후 브라우저에서 해당 서버의 공개 주소로 접속합니다.

## GitHub 배포 주의사항

GitHub에는 프로젝트 소스 코드를 올릴 수 있지만, GitHub Pages는 Java 서버를 실행하지 못합니다.
이 프로젝트는 Java 프로그래밍 과제이므로 GitHub Pages 단독 배포가 아니라 Java 실행이 가능한 웹서버에서 실행해야 합니다.

가능한 배포 방식:

- 개인 서버/VPS/클라우드 서버에 저장소를 clone 한 뒤 `java WorkoutServer` 실행
- 학교/과제용 Java 웹서버에서 실행
- Java 실행을 지원하는 PaaS(Render, Railway 등)에 배포

프론트엔드는 더 이상 특정 로컬 주소를 고정해서 사용하지 않습니다.
로그인과 기록 저장은 현재 접속한 Java 서버의 `/api/login`, `/api/workout`으로 요청됩니다.

## 파일 구성

- `WorkoutServer.java`: Java 웹서버, 로그인/회원가입 API, 운동 기록 저장 API, 정적 파일 서빙
- `index.html`: 메인 UI와 AI 운동 분석 로직
- `coi-serviceworker.js`: WASM 실행 환경 보조 서비스 워커
- `pose_landmarker_full.task`: MediaPipe 포즈 모델
- `static/`: ONNX Runtime Web, MediaPipe WASM 파일
- `models/`: 스쿼트, 푸시업, 풀업 ONNX 모델

## 저장 데이터

회원 정보와 운동 기록은 서버 실행 폴더의 `users_secure.db` 파일에 저장됩니다.
이 파일은 개인정보/비밀번호 해시가 포함될 수 있으므로 GitHub에 커밋하지 않습니다.
