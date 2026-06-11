import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * AI PT Studio - Workout Result Analyzer
 * 자바 프로그래밍 강의 과제용: 파일 입출력 및 객체 지향 설계를 활용한 데이터 분석기
 */
public class WorkoutAnalyzer {

    // 1. 객체 지향 설계를 위한 데이터 클래스 (Class & Encapsulation)
    static class WorkoutSession {
        String exercise;
        int totalReps;
        double avgScore;
        double peakScore;
        List<Double> scores;
        String timestamp;

        public WorkoutSession(String exercise, int totalReps, double avgScore, double peakScore, List<Double> scores, String timestamp) {
            this.exercise = exercise;
            this.totalReps = totalReps;
            this.avgScore = avgScore;
            this.peakScore = peakScore;
            this.scores = scores;
            this.timestamp = timestamp;
        }

        public void printSummary() {
            System.out.println("\n========= 운동 분석 결과 =========");
            System.out.println("운동 종목: " + exercise.toUpperCase());
            System.out.println("실행 시간: " + timestamp);
            System.out.println("--------------------------------");
            System.out.println("총 횟수: " + totalReps + "회");
            System.out.println("평균 점수: " + String.format("%.2f", avgScore) + "점");
            System.out.println("최고 점수: " + String.format("%.2f", peakScore) + "점");
            
            String grade = calculateGrade(avgScore);
            System.out.println("최종 등급: " + grade);
            System.out.println("================================\n");
        }

        private String calculateGrade(double score) {
            if (score >= 90) return "S (완벽한 자세)";
            if (score >= 80) return "A (매우 우수)";
            if (score >= 70) return "B (보통)";
            return "C (자세 교정 필요)";
        }
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.println("AI PT Studio 자바 데이터 분석기를 시작합니다.");
        System.out.print("분석할 JSON 파일 경로를 입력하세요: ");
        String filePath = scanner.nextLine();

        try {
            // 2. 파일 읽기 (File I/O)
            File file = new File(filePath);
            if (!file.exists()) {
                System.out.println("파일을 찾을 수 없습니다: " + filePath);
                return;
            }

            BufferedReader reader = new BufferedReader(new FileReader(file));
            StringBuilder jsonContent = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                jsonContent.append(line);
            }
            reader.close();

            // 3. 데이터 파싱 (자바 기본 문법 활용)
            // 강의 수준에 맞춰 복잡한 라이브러리 없이 문자열 처리로 핵심 데이터 추출
            WorkoutSession session = parseJson(jsonContent.toString());

            if (session != null) {
                session.printSummary();
            }

        } catch (Exception e) {
            System.out.println("파일 분석 중 오류가 발생했습니다: " + e.getMessage());
        }
    }

    // JSON 문자열에서 필요한 데이터를 추출하는 로직 (String manipulation)
    private static WorkoutSession parseJson(String json) {
        try {
            String exercise = extractValue(json, "exercise");
            int totalReps = Integer.parseInt(extractValue(json, "totalReps"));
            double avgScore = Double.parseDouble(extractValue(json, "avgScore"));
            double peakScore = Double.parseDouble(extractValue(json, "peakScore"));
            String timestamp = extractValue(json, "timestamp");

            // 점수 리스트 추출 (간단한 처리)
            List<Double> scores = new ArrayList<>();
            // 실제 구현 시에는 정규식이나 JSON 라이브러리를 쓰지만, 
            // 여기서는 자바 기본기(ArrayList)를 보여주는 것에 집중합니다.

            return new WorkoutSession(exercise, totalReps, avgScore, peakScore, scores, timestamp);
        } catch (Exception e) {
            System.out.println("데이터 파싱 실패: 형식이 올바르지 않습니다.");
            return null;
        }
    }

    private static String extractValue(String json, String key) {
        String searchKey = "\"" + key + "\":";
        int start = json.indexOf(searchKey) + searchKey.length();
        int end = json.indexOf(",", start);
        if (end == -1) end = json.indexOf("}", start);
        
        String value = json.substring(start, end).replace("\"", "").trim();
        return value;
    }
}
