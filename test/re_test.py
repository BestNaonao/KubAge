import re

# 示例文档
text = """
## JSONSchemaProps

JSONSchemaProps 是JSON 模式（JSON-Schema），遵循其规范草案第 4 版（http://json-schema.org/）。

* * *

  * **$ref** (string)

  * **$schema** (string)

  * **additionalItems** (JSONSchemaPropsOrBool)

**JSONSchemaPropsOrBool 表示 JSONSchemaProps 或布尔值。布尔属性默认为 true。**

  * **additionalProperties** (JSONSchemaPropsOrBool)

**JSONSchemaPropsOrBool 表示 JSONSchemaProps 或布尔值。布尔属性默认为 true。**

  * **allOf** ([]JSONSchemaProps)

**原子：将在合并期间被替换**

  * **anyOf** ([]JSONSchemaProps)

**原子：将在合并期间被替换**

  * **default** (JSON)

default 是未定义对象字段的默认值。设置默认值操作是 CustomResourceDefaulting 特性门控所控制的一个 Beta 特性。 应用默认值设置时要求 spec.preserveUnknownFields 为 false。

**JSON 表示任何有效的 JSON 值。支持以下类型：bool、int64、float64、string、[]interface{}、map[string]interface{} 和 nil。**

  * **definitions** (map[string]JSONSchemaProps)

  * **dependencies** (map[string]JSONSchemaPropsOrStringArray)

**JSONSchemaPropsOrStringArray 表示 JSONSchemaProps 或字符串数组。**

  * **description** (string)

  * **enum** ([]JSON)

**原子：将在合并期间被替换**

**JSON 表示任何有效的 JSON 值。支持以下类型：bool、int64、float64、string、[]interface{}、map[string]interface{} 和 nil。**

  * **example** (JSON)

**JSON 表示任何有效的 JSON 值。支持以下类型：bool、int64、float64、string、[]interface{}、map[string]interface{} 和 nil。**

  * **exclusiveMaximum** (boolean)

  * **exclusiveMinimum** (boolean)

  * **externalDocs** (ExternalDocumentation)

**ExternalDocumentation 允许引用外部资源作为扩展文档。**

    * **externalDocs.description** (string)

    * **externalDocs.url** (string)

  * **format** (string)

format 是 OpenAPI v3 格式字符串。未知格式将被忽略。以下格式会被验证合法性：

    * bsonobjectid：一个 bson 对象的 ID，即一个 24 个字符的十六进制字符串
    * uri：由 Go 语言 net/url.ParseRequestURI 解析得到的 URI
    * email：由 Go 语言 net/mail.ParseAddress 解析得到的电子邮件地址
    * hostname：互联网主机名的有效表示，由 RFC 1034 第 3.1 节 [RFC1034] 定义
    * ipv4：由 Go 语言 net.ParseIP 解析得到的 IPv4 协议的 IP
    * ipv6：由 Go 语言 net.ParseIP 解析得到的 IPv6 协议的 IP
    * cidr：由 Go 语言 net.ParseCIDR 解析得到的 CIDR
    * mac：由 Go 语言 net.ParseMAC 解析得到的一个 MAC 地址
    * uuid：UUID，允许大写字母，满足正则表达式 (?i)^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$
    * uuid3：UUID3，允许大写字母，满足正则表达式 (?i)^[0-9a-f]{8}-?[0-9a-f]{4}-?3[0-9a-f]{3}-?[0-9a-f]{4}-?[0-9a-f]{12}$
    * uuid4：UUID4，允许大写字母，满足正则表达式 (?i)^[0-9a-f]{8}-?[0-9a-f]{4}-?4[0-9a-f]{3}-?[89ab][0-9a-f]{3}-?[0-9a-f]{12}$
    * uuid5：UUID5，允许大写字母，满足正则表达式 (?i)^[0-9a-f]{8}-?[0-9a-f]{4}-?5[0-9a-f]{3}-?[89ab][0-9a-f]{3}-?[0-9a-f]{12}$
    * isbn：一个 ISBN10 或 ISBN13 数字字符串，如 "0321751043" 或 "978-0321751041"
    * isbn10：一个 ISBN10 数字字符串，如 "0321751043"
    * isbn13：一个 ISBN13 号码字符串，如 "978-0321751041"
    * creditcard：信用卡号码，满足正则表达式 ^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|6(?:011|5[0-9][0-9])[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|(?:2131|1800|35\d{3})\d{11})$， 其中混合任意非数字字符
    * ssn：美国社会安全号码，满足正则表达式 ^\d{3}[- ]?\d{2}[- ]?\d{4}$
    * hexcolor：一个十六进制的颜色编码，如 "#FFFFFF"，满足正则表达式 ^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$
    * rgbcolor：一个 RGB 颜色编码 例如 "rgb(255,255,255)"
    * byte：base64 编码的二进制数据
    * password：任何类型的字符串
    * date：类似 "2006-01-02" 的日期字符串，由 RFC3339 中的完整日期定义
    * duration：由 Go 语言 time.ParseDuration 解析的持续时长字符串，如 "22 ns"，或与 Scala 持续时间格式兼容。
    * datetime：一个日期时间字符串，如 "2014-12-15T19:30:20.000Z"，由 RFC3339 中的 date-time 定义。
  * **id** (string)

  * **items** (JSONSchemaPropsOrArray)

**JSONSchemaPropsOrArray 表示可以是 JSONSchemaProps 或 JSONSchemaProps 数组的值。这里目的主要用于序列化。**

  * **maxItems** (int64)

  * **maxLength** (int64)

  * **maxProperties** (int64)

  * **maximum** (double)

  * **minItems** (int64)

  * **minLength** (int64)

  * **minProperties** (int64)

  * **minimum** (double)

  * **multipleOf** (double)

  * **not** (JSONSchemaProps)

  * **nullable** (boolean)

  * **oneOf** ([]JSONSchemaProps)

**原子：将在合并期间被替换**

  * **pattern** (string)

  * **patternProperties** (map[string]JSONSchemaProps)

  * **properties** (map[string]JSONSchemaProps)

  * **required** ([]string)

**原子：将在合并期间被替换**

  * **title** (string)

  * **type** (string)

  * **uniqueItems** (boolean)

  * **x-kubernetes-embedded-resource** (boolean)

x-kubernetes-embedded-resource 定义该值是一个嵌入式 Kubernetes runtime.Object，具有 TypeMeta 和 ObjectMeta。 类型必须是对象。允许进一步限制嵌入对象。会自动验证 kind、apiVersion 和 metadata 等字段值。 x-kubernetes-preserve-unknown-fields 允许为 true，但如果对象已完全指定 （除 kind、apiVersion、metadata 之外），则不必为 true。

  * **x-kubernetes-int-or-string** (boolean)

x-kubernetes-int-or-string 指定此值是整数或字符串。如果为 true，则允许使用空类型， 并且如果遵循以下模式之一，则允许作为 anyOf 的子类型：

    1. anyOf:
       * type: integer
       * type: string
    2. allOf:
       * anyOf:
         * type: integer
         * type: string
       * （可以有选择地包含其他类型）
  * **x-kubernetes-list-map-keys** ([]string)

**原子：将在合并期间被替换**

X-kubernetes-list-map-keys 通过指定用作 map 索引的键来使用 x-kubernetes-list-type `map` 注解数组。

这个标签必须只用于 "x-kubernetes-list-type" 扩展设置为 "map" 的列表。 而且，为这个属性指定的值必须是子结构的标量类型的字段（不支持嵌套）。

指定的属性必须是必需的或具有默认值，以确保所有列表项都存在这些属性。

  * **x-kubernetes-list-type** (string)

x-kubernetes-list-type 注解一个数组以进一步描述其拓扑。此扩展名只能用于列表，并且可能有 3 个可能的值：

    1. `atomic`： 列表被视为单个实体，就像标量一样。原子列表在更新时将被完全替换。这个扩展可以用于任何类型的列表（结构，标量，…）。
    2. `set`： set 是不能有多个具有相同值的列表。每个值必须是标量、具有 x-kubernetes-map-type `atomic` 的对象或具有 x-kubernetes-list-type `atomic` 的数组。
    3. `map`： 这些列表类似于映射表，因为它们的元素具有用于标识它们的非索引键。合并时保留顺序。 map 标记只能用于元数类型为 object 的列表。 数组默认为原子数组。
  * **x-kubernetes-map-type** (string)

x-kubernetes-map-type 注解一个对象以进一步描述其拓扑。此扩展只能在 type 为 object 时使用，并且可能有 2 个可能的值：

    1. `granular`： 这些 map 是真实的映射（键值对），每个字段都是相互独立的（它们都可以由不同的角色来操作）。 这是所有 map 的默认行为。
    2. `atomic`：map 被视为单个实体，就像标量一样。原子 map 更新后将被完全替换。
  * **x-kubernetes-preserve-unknown-fields** (boolean)

x-kubernetes-preserve-unknown-fields 针对未在验证模式中指定的字段，禁止 API 服务器的解码步骤剪除这些字段。 这一设置对字段的影响是递归的，但在模式中指定了嵌套 properties 或 additionalProperties 时，会切换回正常的字段剪除行为。 该值可为 true 或 undefined，不能为 false。

  * **x-kubernetes-validations** ([]ValidationRule)

**补丁策略：基于键`rule` 合并**

**Map：合并时将保留 rule 键的唯一值**

x-kubernetes-validations 描述了用 CEL 表达式语言编写的验证规则列表。此字段是 Alpha 级别。

**ValidationRule 描述用 CEL 表达式语言编写的验证规则。**

    * **x-kubernetes-validations.rule** (string)，必需

rule 表示将由 CEL 评估的表达式。参考： https://github.com/google/cel-spec。 rule 的作用域为模式中的 x-kubernetes-validation 扩展所在的位置。CEL 表达式中的 `self` 与作用域值绑定。 例子：rule 的作用域是一个具有状态子资源的资源根：{"rule": "self.status.actual <= self.spec.maxDesired"}。

如果 rule 的作用域是一个带有属性的对象，那么该对象的可访问属性是通过 `self` 进行字段选择的， 并且可以通过 `has(self.field)` 来检查字段是否存在。在 CEL 表达式中，Null 字段被视为不存在的字段。 如果该 rule 的作用域是一个带有附加属性的对象（例如一个 map），那么该 map 的值可以通过 `self[mapKey]`来访问，map 是否包含某主键可以通过 `mapKey in self` 来检查。 map 中的所有条目都可以通过 CEL 宏和函数（如 `self.all(...)`）访问。 如果 rule 的作用域是一个数组，数组的元素可以通过 `self[i]` 访问，也可以通过宏和函数访问。 如果 rule 的作用域为标量，`self` 绑定到标量值。举例：

      * rule 作用域为对象映射：{"rule": "self.components['Widget'].priority < 10"}
      * rule 作用域为整数列表：{"rule": "self.values.all(value, value >= 0 && value < 100)"}
      * rule 作用域为字符串值：{"rule": "self.startsWith('kube')"}

`apiVersion`、`kind`、`metadata.name` 和 `metadata.generateName` 总是可以从对象的根和任何带 x-kubernetes-embedded-resource 注解的对象访问。其他元数据属性都无法访问。

在 CEL 表达式中无法访问通过 x-kubernetes-preserve-unknown-fields 保存在自定义资源中的未知数据。 这包括：

      * 由包含 x-kubernetes-preserve-unknown-fields 的对象模式所保留的未知字段值；

      * 属性模式为 "未知类型" 的对象属性。"未知类型" 递归定义为：

        * 没有设置 type 但 x-kubernetes-preserve-unknown-fields 设置为 true 的模式。
        * 条目模式为"未知类型"的数组。
        * additionalProperties 模式为"未知类型"的对象。

只有名称符合正则表达式 `[a-zA-Z_.-/][a-zA-Z0-9_.-/]*` 的属性才可被访问。 在表达式中访问属性时，可访问的属性名称根据以下规则进行转义：

      * '__' 转义为 '**underscores** '

      * '.' 转义为 '**dot** '

      * '-' 转义为 '**dash** '

      * '/' 转义为 '**slash** '

      * 恰好匹配 CEL 保留关键字的属性名称转义为 '**{keyword}** ' 。这里的关键字具体包括： "true"，"false"，"null"，"in"，"as"，"break"，"const"，"continue"，"else"，"for"，"function"，"if"， "import"，"let"，"loop"，"package"，"namespace"，"return"。 举例：

        * 规则访问名为 "namespace" 的属性：`{"rule": "self.__namespace__ > 0"}`
        * 规则访问名为 "x-prop" 的属性：`{"rule": "self.x__dash__prop > 0"}`
        * 规则访问名为 "redact__d" 的属性：`{"rule": "self.redact__underscores__d > 0"}`

对 x-kubernetes-list-type 为 'set' 或 'map' 的数组进行比较时忽略元素顺序，如：[1, 2] == [2, 1]。 使用 x-kubernetes-list-type 对数组进行串接使用下列类型的语义：

      * 'set'：`X + Y` 执行合并，其中 `X` 保留所有元素的数组位置，并附加不相交的元素 `Y`，保留其局部顺序。
      * 'map'：`X + Y` 执行合并，保留 `X` 中所有键的数组位置，但当 `X` 和 `Y` 的键集相交时，会被 `Y` 中的值覆盖。 添加 `Y` 中具有不相交键的元素，保持其局顺序。

如果 `rule` 使用 `oldSelf` 变量，则隐式地将其视为一个 `转换规则（transition rule）`。

默认情况下，`oldSelf` 变量与 `self` 类型相同。当 `optionalOldSelf` 为 `true` 时，`oldSelf` 变量是 CEL 可选变量，其 `value()` 与 `self` 类型相同。 有关详细信息，请参阅 `optionalOldSelf` 字段的文档。

默认情况下，转换规则仅适用于 UPDATE 请求，如果找不到旧值，则会跳过转换规则。 你可以通过将 `optionalOldSelf` 设置为 `true` 来使转换规则进行无条件求值。

    * **x-kubernetes-validations.fieldPath** (string)

fieldPath 表示验证失败时返回的字段路径。 它必须是相对 JSON 路径（即，支持数组表示法），范围仅限于此 x-kubernetes-validations 扩展在模式的位置，并引用现有字段。 例如，当验证检查 `testMap` 映射下是否有 `foo` 属性时，可以将 fieldPath 设置为 `.testMap.foo`。 如果验证需要确保两个列表具有各不相同的属性，则可以将 fieldPath 设置到其中任一列表，例如 `.testList`。 它支持使用子操作引用现有字段，而不支持列表的数字索引。 有关更多信息，请参阅 Kubernetes 中的 JSONPath 支持。 因为其不支持数组的数字索引，所以对于包含特殊字符的字段名称，请使用 `['specialName']` 来引用字段名称。 例如，对于出现在列表 `testList` 中的属性 `foo.34$`，fieldPath 可以设置为 `.testList['foo.34$']`。

    * **x-kubernetes-validations.message** (string)

message 表示验证失败时显示的消息。如果规则包含换行符，则需要该消息。消息不能包含换行符。 如果未设置，则消息为 "failed rule: {Rule}"，如："must be a URL with the host matching spec.host"

    * **x-kubernetes-validations.messageExpression** (string)

messageExpression 声明一个 CEL 表达式，其计算结果是此规则失败时返回的验证失败消息。 由于 messageExpression 用作失败消息，因此它的值必须是一个字符串。 如果在规则中同时存在 message 和 messageExpression，则在验证失败时使用 messageExpression。 如果是 messageExpression 出现运行时错误，则会记录运行时错误，并生成验证失败消息， 就好像未设置 messageExpression 字段一样。如果 messageExpression 求值为空字符串、 只包含空格的字符串或包含换行符的字符串，则验证失败消息也将像未设置 messageExpression 字段一样生成， 并记录 messageExpression 生成空字符串/只包含空格的字符串/包含换行符的字符串的事实。 messageExpression 可以访问的变量与规则相同；唯一的区别是返回类型。 例如："x must be less than max ("+string(self.max)+")"。

    * **x-kubernetes-validations.optionalOldSelf** (boolean)

即使在对象首次创建时，或者旧对象无值时，也可以使用 `optionalOldSelf` 来使用转换规则求值。

当启用了 `optionalOldSelf` 时，`oldSelf` 将是 CEL 可选项，如果没有旧值或最初创建对象时，其值将为 `None`。

你可以使用 `oldSelf.hasValue()` 检查 oldSelf 是否存在，并在检查后使用 `oldSelf.value()` 将其解包。 更多的信息可查看 CEL 文档中的 Optional 类型：https://pkg.go.dev/github.com/google/cel-go/cel#OptionalTypes

除非在 `rule` 中使用了 `oldSelf`，否则不可以设置。

    * **x-kubernetes-validations.reason** (string)

`reason` 提供机器可读的验证失败原因，当请求未通过此验证规则时，该原因会返回给调用者。 返回给调用者的 HTTP 状态代码将与第一个失败的验证规则的原因相匹配。 目前支持的原因有：`FieldValueInvalid`、`FieldValueForbidden`、`FieldValueRequired`、`FieldValueDuplicate`。 如果未设置，则默认使用 `FieldValueInvalid`。 所有未来添加的原因在读取该值时必须被客户端接受，未知原因应被视为 `FieldValueInvalid`。

可能的枚举值：

      * `"FieldValueDuplicate"` 用于报告取值必须唯一的值之间的冲突（例如，唯一 ID）。
      * `"FieldValueForbidden"` 用于报告在某些条件下可接受（根据格式规则），但当前条件下不允许的合法值（例如，安全策略）。
      * `"FieldValueInvalid"` 用于报告格式错误的值（例如，正则表达式匹配失败、过长、越界）。
      * `"FieldValueRequired"` 用于报告未提供的必需值（例如，空字符串、null 值或空数组）。



"""

# 正则表达式
pattern = r'\n\*\*(.+?)\*\*(?:\n(?!\*\*)(.*?))?(?=\n\*\*|\Z)'

# 使用 re.DOTALL 让 . 匹配换行符
matches = re.findall(pattern, text, re.DOTALL)

# 输出结果
for i, (title, description) in enumerate(matches, 1):
    print(f"小标题 {i}: {title}")
    print(f"描述: {description.strip() if description else '(无描述)'}")
    print("-" * 30)