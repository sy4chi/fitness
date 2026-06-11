import com.sun.net.httpserver.HttpServer;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpExchange;
import java.io.*;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.*;

/**
 * [Security Expert Mode] AI PT Studio Backend
 * Separated Login and Registration to prevent ID duplication.
 */
public class WorkoutServer {
    private static final String USER_DB = "users_secure.db";
    private static final int DEFAULT_PORT = 8080;
    private static final Path WEB_ROOT = Paths.get(".").toAbsolutePath().normalize();

    public static void main(String[] args) throws IOException {
        int port = getPort();
        HttpServer server = HttpServer.create(new InetSocketAddress("0.0.0.0", port), 0);
        server.createContext("/api/login", new LoginHandler());
        server.createContext("/api/workout", new WorkoutHandler());
        server.createContext("/api/ranking", new RankingHandler());
        server.createContext("/", new StaticFileHandler());
        server.setExecutor(null);
        System.out.println("🛡️ [Security Expert Mode] 서버가 시작되었습니다.");
        System.out.println("🔒 로그인/회원가입 분리 모드 활성화 (ID 중복 방지)");
        System.out.println("🌐 웹 앱 포트: " + port);
        server.start();
    }

    private static int getPort() {
        String port = System.getenv("PORT");
        if (port == null || port.isBlank()) return DEFAULT_PORT;
        try {
            return Integer.parseInt(port);
        } catch (NumberFormatException e) {
            return DEFAULT_PORT;
        }
    }

    private static String hashPassword(String password) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] encodedhash = digest.digest(password.getBytes(StandardCharsets.UTF_8));
            StringBuilder hexString = new StringBuilder(2 * encodedhash.length);
            for (byte b : encodedhash) {
                String hex = Integer.toHexString(0xff & b);
                if (hex.length() == 1) hexString.append('0');
                hexString.append(hex);
            }
            return hexString.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
    }

    static class LoginHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            setupCORS(exchange);
            if (exchange.getRequestMethod().equalsIgnoreCase("OPTIONS")) { exchange.sendResponseHeaders(204, -1); return; }

            if (exchange.getRequestMethod().equalsIgnoreCase("POST")) {
                String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
                String decoded = new String(Base64.getDecoder().decode(body), StandardCharsets.UTF_8);
                
                // 형식: "userId:password:action"
                String[] parts = decoded.split(":", 3);
                if (parts.length < 3) {
                    sendResponse(exchange, 400, "{\"status\":\"fail\", \"message\":\"잘못된 요청 형식\"}");
                    return;
                }

                String id = parts[0];
                String hashedPw = hashPassword(parts[1]);
                String action = parts[2];

                if ("register".equals(action)) {
                    if (isUserExist(id)) {
                        sendResponse(exchange, 409, "{\"status\":\"fail\", \"message\":\"이미 사용 중인 아이디입니다.\"}");
                    } else {
                        registerUser(id, hashedPw);
                        sendResponse(exchange, 200, "{\"status\":\"success\", \"message\":\"회원가입이 완료되었습니다! 로그인 해주세요.\"}");
                    }
                } else { // login
                    String userData = findUser(id, hashedPw);
                    if (userData != null) {
                        sendResponse(exchange, 200, "{\"status\":\"success\", \"message\":\"로그인 성공!\", \"data\":" + userData + "}");
                    } else {
                        if (isUserExist(id)) {
                            sendResponse(exchange, 401, "{\"status\":\"fail\", \"message\":\"비밀번호가 틀렸습니다.\"}");
                        } else {
                            sendResponse(exchange, 404, "{\"status\":\"fail\", \"message\":\"존재하지 않는 아이디입니다.\"}");
                        }
                    }
                }
            } else {
                sendResponse(exchange, 405, "{\"status\":\"fail\", \"message\":\"POST 요청만 허용됩니다.\"}");
            }
        }
    }

    static class WorkoutHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            setupCORS(exchange);
            if (exchange.getRequestMethod().equalsIgnoreCase("OPTIONS")) { exchange.sendResponseHeaders(204, -1); return; }

            if (exchange.getRequestMethod().equalsIgnoreCase("POST")) {
                String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
                String decoded = new String(Base64.getDecoder().decode(body), StandardCharsets.UTF_8);
                
                if (validateData(decoded)) {
                    updateUserData(decoded);
                    sendResponse(exchange, 200, "{\"status\":\"success\"}");
                } else {
                    sendResponse(exchange, 400, "{\"status\":\"error\", \"message\":\"Invalid Data\"}");
                }
            } else {
                sendResponse(exchange, 405, "{\"status\":\"error\", \"message\":\"POST 요청만 허용됩니다.\"}");
            }
        }
        
        private boolean validateData(String json) {
            try {
                double avgScore = Double.parseDouble(extractValue(json, "avgScore"));
                return avgScore >= 0 && avgScore <= 100;
            } catch (Exception e) { return false; }
        }
    }


    static class RankingHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            setupCORS(exchange);
            if (exchange.getRequestMethod().equalsIgnoreCase("OPTIONS")) { exchange.sendResponseHeaders(204, -1); return; }

            if (!exchange.getRequestMethod().equalsIgnoreCase("GET")) {
                sendResponse(exchange, 405, "{\"status\":\"error\", \"message\":\"GET 요청만 허용됩니다.\"}");
                return;
            }

            List<UserRank> ranking = readRanking();
            ranking.sort((a, b) -> {
                int scoreCompare = Double.compare(b.avgScore, a.avgScore);
                if (scoreCompare != 0) return scoreCompare;
                int peakCompare = Double.compare(b.peakScore, a.peakScore);
                if (peakCompare != 0) return peakCompare;
                return Integer.compare(b.totalReps, a.totalReps);
            });

            StringBuilder json = new StringBuilder();
            json.append("{\"status\":\"success\",\"ranking\":[");
            for (int i = 0; i < ranking.size(); i++) {
                UserRank r = ranking.get(i);
                if (i > 0) json.append(",");
                json.append("{")
                    .append("\"id\":\"").append(escapeJson(r.id)).append("\",")
                    .append("\"userId\":\"").append(escapeJson(r.id)).append("\",")
                    .append("\"avgScore\":").append(formatNumber(r.avgScore)).append(",")
                    .append("\"peakScore\":").append(formatNumber(r.peakScore)).append(",")
                    .append("\"totalReps\":").append(r.totalReps).append(",")
                    .append("\"exercise\":\"").append(escapeJson(r.exercise)).append("\",")
                    .append("\"timestamp\":\"").append(escapeJson(r.timestamp)).append("\"")
                    .append("}");
            }
            json.append("]}");
            sendResponse(exchange, 200, json.toString());
        }
    }

    static class UserRank {
        String id;
        double avgScore;
        double peakScore;
        int totalReps;
        String exercise;
        String timestamp;

        UserRank(String id, String data) {
            this.id = id;
            this.avgScore = parseDoubleSafe(extractValueSafe(data, "avgScore"));
            this.peakScore = parseDoubleSafe(extractValueSafe(data, "peakScore"));
            this.totalReps = parseIntSafe(extractValueSafe(data, "totalReps"));
            this.exercise = extractValueSafe(data, "exercise");
            this.timestamp = extractValueSafe(data, "timestamp");
        }
    }

    static class StaticFileHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            boolean isHead = exchange.getRequestMethod().equalsIgnoreCase("HEAD");
            if (!exchange.getRequestMethod().equalsIgnoreCase("GET") && !isHead) {
                sendText(exchange, 405, "Method Not Allowed", "text/plain; charset=utf-8");
                return;
            }

            String requestPath = URLDecoder.decode(exchange.getRequestURI().getPath(), StandardCharsets.UTF_8);
            if (requestPath.equals("/")) requestPath = "/index.html";

            Path filePath = WEB_ROOT.resolve(requestPath.substring(1)).normalize();
            if (!filePath.startsWith(WEB_ROOT) || !Files.isRegularFile(filePath)) {
                sendText(exchange, 404, "Not Found", "text/plain; charset=utf-8");
                return;
            }

            byte[] bytes = Files.readAllBytes(filePath);
            addWebHeaders(exchange);
            exchange.getResponseHeaders().set("Content-Type", getContentType(filePath));
            if (isHead) {
                exchange.sendResponseHeaders(200, -1);
                exchange.close();
                return;
            }
            exchange.sendResponseHeaders(200, bytes.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(bytes);
            }
        }
    }

    private static String getContentType(Path filePath) {
        String name = filePath.getFileName().toString().toLowerCase(Locale.ROOT);
        if (name.endsWith(".html")) return "text/html; charset=utf-8";
        if (name.endsWith(".js") || name.endsWith(".mjs")) return "text/javascript; charset=utf-8";
        if (name.endsWith(".css")) return "text/css; charset=utf-8";
        if (name.endsWith(".json")) return "application/json; charset=utf-8";
        if (name.endsWith(".wasm")) return "application/wasm";
        if (name.endsWith(".onnx")) return "application/octet-stream";
        if (name.endsWith(".task")) return "application/octet-stream";
        if (name.endsWith(".png")) return "image/png";
        if (name.endsWith(".jpg") || name.endsWith(".jpeg")) return "image/jpeg";
        if (name.endsWith(".svg")) return "image/svg+xml";
        return "application/octet-stream";
    }

    private static synchronized String findUser(String id, String hashedPw) {
        try (BufferedReader br = new BufferedReader(new FileReader(USER_DB))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split("\\|", 3);
                if (parts[0].equals(id) && parts[1].equals(hashedPw)) return parts[2];
            }
        } catch (IOException e) { }
        return null;
    }

    private static boolean isUserExist(String id) {
        try (BufferedReader br = new BufferedReader(new FileReader(USER_DB))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.split("\\|")[0].equals(id)) return true;
            }
        } catch (IOException e) { }
        return false;
    }

    private static synchronized void registerUser(String id, String hashedPw) {
        String initialData = "{\"totalReps\":0, \"avgScore\":0.0}";
        try (PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(USER_DB, true)))) {
            out.println(id + "|" + hashedPw + "|" + initialData);
        } catch (IOException e) { e.printStackTrace(); }
    }

    private static synchronized void updateUserData(String json) {
        String id = extractValue(json, "userId");
        List<String> lines = new ArrayList<>();
        try {
            File file = new File(USER_DB);
            if (file.exists()) {
                try (BufferedReader br = new BufferedReader(new FileReader(file))) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        String[] parts = line.split("\\|", 3);
                        if (parts[0].equals(id)) {
                            lines.add(id + "|" + parts[1] + "|" + json);
                        } else {
                            lines.add(line);
                        }
                    }
                }
                try (PrintWriter pwOut = new PrintWriter(new FileWriter(USER_DB))) {
                    for (String l : lines) pwOut.println(l);
                }
            }
        } catch (IOException e) { }
    }


    private static synchronized List<UserRank> readRanking() {
        List<UserRank> ranking = new ArrayList<>();
        File file = new File(USER_DB);
        if (!file.exists()) return ranking;

        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split("\\|", 3);
                if (parts.length >= 3) {
                    ranking.add(new UserRank(parts[0], parts[2]));
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return ranking;
    }

    private static String extractValueSafe(String json, String key) {
        if (json == null || key == null) return "";
        String search = "\"" + key + "\":";
        int start = json.indexOf(search);
        if (start < 0) return "";
        start += search.length();
        while (start < json.length() && Character.isWhitespace(json.charAt(start))) start++;

        if (start < json.length() && json.charAt(start) == '\"') {
            start++;
            StringBuilder value = new StringBuilder();
            boolean escaped = false;
            for (int i = start; i < json.length(); i++) {
                char c = json.charAt(i);
                if (escaped) {
                    value.append(c);
                    escaped = false;
                } else if (c == '\\') {
                    escaped = true;
                } else if (c == '\"') {
                    break;
                } else {
                    value.append(c);
                }
            }
            return value.toString().trim();
        }

        int end = json.indexOf(",", start);
        if (end == -1) end = json.indexOf("}", start);
        if (end == -1) end = json.length();
        return json.substring(start, end).replace("\"", "").trim();
    }

    private static double parseDoubleSafe(String value) {
        try { return Double.parseDouble(value); }
        catch (Exception e) { return 0.0; }
    }

    private static int parseIntSafe(String value) {
        try { return (int)Math.round(Double.parseDouble(value)); }
        catch (Exception e) { return 0; }
    }

    private static String formatNumber(double value) {
        return String.format(Locale.US, "%.1f", value);
    }

    private static String escapeJson(String value) {
        if (value == null) return "";
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static String extractValue(String json, String key) {
        String search = "\"" + key + "\":";
        int start = json.indexOf(search) + search.length();
        if (json.charAt(start) == '\"') start++;
        int end = json.indexOf(",", start);
        if (end == -1) end = json.indexOf("}", start);
        return json.substring(start, end).replace("\"", "").trim();
    }

    private static void setupCORS(HttpExchange exchange) {
        exchange.getResponseHeaders().add("Access-Control-Allow-Origin", "*");
        exchange.getResponseHeaders().add("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        exchange.getResponseHeaders().add("Access-Control-Allow-Headers", "Content-Type");
        addWebHeaders(exchange);
    }

    private static void addWebHeaders(HttpExchange exchange) {
        exchange.getResponseHeaders().set("Cross-Origin-Opener-Policy", "same-origin");
        exchange.getResponseHeaders().set("Cross-Origin-Embedder-Policy", "require-corp");
        exchange.getResponseHeaders().set("Cross-Origin-Resource-Policy", "same-origin");
    }

    private static void sendResponse(HttpExchange exchange, int status, String response) throws IOException {
        sendText(exchange, status, response, "application/json; charset=utf-8");
    }

    private static void sendText(HttpExchange exchange, int status, String response, String contentType) throws IOException {
        byte[] bytes = response.getBytes(StandardCharsets.UTF_8);
        addWebHeaders(exchange);
        exchange.getResponseHeaders().set("Content-Type", contentType);
        exchange.sendResponseHeaders(status, bytes.length);
        OutputStream os = exchange.getResponseBody();
        os.write(bytes);
        os.close();
    }
}
