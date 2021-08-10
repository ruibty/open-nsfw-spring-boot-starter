# open-nsfw-spring-boot-starter
[![](https://img.shields.io/badge/Download-1.0-brightgreen.svg)](https://github.com/ruibty/open-nsfw-spring-boot-starter)
[![](https://img.shields.io/badge/Base-TensorFlow-green.svg)](https://github.com/ruibty/open-nsfw-spring-boot-starter)
[![](https://img.shields.io/badge/license-Apache%202-orange.svg)](https://www.apache.org/licenses/LICENSE-2.0)

本地内置训练好的模型，结果以不大于1的小数表示其内容的不安全可能性。

结果受限于模型和技术原理，不能保证百分百的准确性。

## base
- https://github.com/yahoo/open_nsfw
- https://github.com/tensorflow/tensorflow

## 使用

1. pom.xml
```
<!-- https://mvnrepository.com/artifact/com.ruibty.nsfw/open-nsfw-spring-boot-starter -->
<dependency>
    <groupId>com.ruibty.nsfw</groupId>
    <artifactId>open-nsfw-spring-boot-starter</artifactId>
    <version>1.0</version>
</dependency>
```
2. eg.
```
@RestController
@SpringBootApplication
public class DemoApplication {
    public static void main(String[] args) {
        SpringApplication.run(DemoApplication.class, args);
    }

    @Autowired
    private NsfwService nsfwService;

    @PostMapping
    public ResponseEntity<BigDecimal> upload(
            @RequestParam("file") MultipartFile multipartFile
    ) throws IOException {
        byte[] imgBytes = multipartFile.getBytes();
        float prediction = nsfwService.getPrediction(imgBytes);
        BigDecimal result = new BigDecimal(String.valueOf(prediction));
        return ResponseEntity.ok(result);
    }
}
```

## 联系方式

如果有bug、讨论、新功能建议等等，欢迎你找到我：

- 邮箱
<a href="mailto:ruibty@qq.com?subject=Storage存储&body=你好">
ruibty@qq.com
</a>
