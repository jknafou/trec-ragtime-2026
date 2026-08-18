"""Versioned boilerplate rule sets for the chunker's structural strip.

The chunker splits a document into paragraphs on blank-line and structural boundaries and,
when ``strip_boilerplate`` is on, drops pure-boilerplate lines (nav breadcrumbs, cookie and
consent banners, social-share widgets, standalone datelines and bylines, copyright footers,
"read more" and pagination chrome) before sentence segmentation. The rules are a versioned
rule set in code: the config carries only ``boilerplate_rules: "v1"``, a small string that
folds into the ``chunker`` block's hash, and :func:`boilerplate_rules` resolves it to the
concrete matcher. Bumping the rule set is an addition here plus a one-line config flip,
never a change to the config's shape.

Precision is preferred to recall throughout. Every rule fires only when the whole stripped
line is chrome (short, standalone, matching a specific shape) and never when a boilerplate
keyword merely appears inside a line of real prose: missing some chrome is acceptable,
deleting a real sentence is not. A breadcrumb needs the actual ``a > b > c`` path shape, so
a bare ``Home /`` never fires; a copyright footer needs a copyright marker together with
either "all rights reserved" or a year; a cookie line needs the banner call-to-action
structure rather than any sentence mentioning cookies. All rules are additionally gated to
short lines.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "BOILERPLATE_RULES_V1",
    "BOILERPLATE_RULES_V2",
    "boilerplate_rules",
    "is_boilerplate",
]

# Chrome is short; a longer line is treated as prose and never stripped, which guards a
# real sentence that merely contains a chrome word ("...violated copyright law...").
_MAX_CHROME_LEN = 120

_Rule = tuple[str, "Callable[[str], bool]"]


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.UNICODE)


# Nav breadcrumbs: the `a > b > c` path shape, needing at least two chevron separators
# between at least three short label segments. `/` and `|` are excluded because they appear
# in prose and URLs, so "Home > News" and "Home / ..." never fire.
_CHEVRON = re.compile(r"[›»>]")


def _is_breadcrumb(line: str) -> bool:
    parts = [p.strip() for p in _CHEVRON.split(line)]
    parts = [p for p in parts if p]
    if len(parts) < 3:  # two separators give three non-empty segments
        return False
    return all(len(p) <= 30 for p in parts)


# Cookie and consent banners: the banner call-to-action structure as a whole short line,
# either an imperative verb wrapping "cookies" or a "we use cookies" lead-in. A prose
# sentence that merely mentions cookies is not a call to action and never fires.
_COOKIE_CTA = _rx(
    r"^(accept|allow|reject|decline|manage|customi[sz]e|enable|disable|got it|"
    r"i accept|i agree|agree|okay|ok)\b[\w\s,'’.&\-]*\bcookies?\b[\w\s,'’.&\-]*$"
)
_COOKIE_LEAD = _rx(r"^(this (site|website) uses cookies|we use cookies)\b.*$")


def _is_cookie(line: str) -> bool:
    return bool(_COOKIE_CTA.match(line) or _COOKIE_LEAD.match(line))


# Social-share widgets: a whole line of two or more share or platform tokens joined by
# delimiters, or a strict "share this <thing>" / "share on <platform>" / "share via
# <channel>" lead-in. "Share the wealth among residents" does not fire.
_SHARE_WORD = (
    r"(?:share|tweet|facebook|whats\s?app|e-?mail|print|linkedin|reddit|pinterest|"
    r"telegram|messenger|flipboard|copy link|compartir|tuitear|поделиться|分享)"
)
_SHARE_LINE = _rx(rf"^{_SHARE_WORD}(?:[\s|/·•,–—-]+{_SHARE_WORD})+[.!]?$")
_SHARE_LEAD = _rx(
    r"^share (this (article|story|post|page|photo|video)|"
    r"on (facebook|twitter|x|whatsapp|linkedin|reddit)|"
    r"via (email|e-?mail|whatsapp))\b.{0,40}$"
)


def _is_share(line: str) -> bool:
    return bool(_SHARE_LINE.match(line) or _SHARE_LEAD.match(line))


# Standalone bylines and datelines, whole short line only. In a byline every token after
# the lead must be capitalised, so "By Monday the deal was done." is not one.
_BYLINE = _rx(
    r"^(by|por|von)\s+[A-ZА-ЯÀ-Þ][\w.'’\-]*(?:\s+[A-ZА-ЯÀ-Þ][\w.'’\-]*){0,3}\.?$"
)
_MONTH = (
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|ene|abr|ago|dic|"
    r"january|february|march|april|june|july|august|september|october|november|december)"
)
# "CITY, Month DD, YYYY" datelines and standalone agency tags.
_DATELINE = _rx(
    rf"^[A-ZÀ-Þ][A-ZÀ-Þ .'\-]{{1,39}},?\s+{_MONTH}[a-z.]*\s+\d{{1,2}},?\s+\d{{4}}\.?$"
)
_AGENCY = _rx(r"^\(?(reuters|ap|afp|efe|tass|xinhua|bloomberg|associated press)\)?\.?$")


def _is_dateline(line: str) -> bool:
    return bool(_BYLINE.match(line) or _DATELINE.match(line) or _AGENCY.match(line))


# Copyright footers: an explicit "all rights reserved" phrase, which is footer-defining on
# its own, or a copyright marker together with a four-digit year. A prose line mentioning
# "copyright law", or a clause label "(c)", never fires.
_ARR = _rx(
    r"all rights reserved|todos los derechos reservados|derechos reservados|"
    r"все права защищены|版权所有"
)
_COPY_SYMBOL = _rx(r"©|&copy;|\bcopyright\b|\(c\)\s*(?:19|20)\d{2}")
_YEAR = _rx(r"\b(?:19|20)\d{2}\b")


def _is_copyright(line: str) -> bool:
    return bool(_ARR.search(line)) or (
        bool(_COPY_SYMBOL.search(line)) and bool(_YEAR.search(line))
    )


# "Read more", pagination and newsletter chrome; whole short line, anchored.
_READMORE = _rx(
    r"^(read more|continue reading|read the full (story|article)|leer m[áa]s|"
    r"читать далее|阅读更多|next|previous|prev|newer|older|load more|show more|"
    r"see more|view more|more stories|related (articles|stories)|advertisement|"
    r"sponsored( content)?|subscribe|sign up|newsletter|follow us)\b.{0,40}$"
)
_PAGINATION = _rx(r"^(page\s+\d+(\s+of\s+\d+)?|(?:\d+\s+){2,}(next|older|»)?)$")


def _is_readmore(line: str) -> bool:
    return bool(_READMORE.match(line) or _PAGINATION.match(line))


# The v1 rule set: (category-name, predicate). The category names are diagnostic only.
BOILERPLATE_RULES_V1: tuple[_Rule, ...] = (
    ("nav_breadcrumb", _is_breadcrumb),
    ("cookie_consent", _is_cookie),
    ("social_share", _is_share),
    ("dateline_byline", _is_dateline),
    ("copyright_footer", _is_copyright),
    ("read_more_pagination", _is_readmore),
)

# Rule set v2: multilingual, corpus-mined additions for zh, ru, es and en. v1 is left
# untouched. The entries come from a chrome-mining pass over the real corpus (repeated
# exact stripped lines, ranked by document frequency); v1 was English-centric and left
# non-English nav chrome in the passages verbatim. v2 adds three things:
#   (a) an exact whole-line inventory of mined chrome, matched under a light
#       normalization so "相關新聞", "相關新聞：" and "• 相關新聞" are one key;
#   (b) a few tight anchored patterns generalising a mined family (comment counters,
#       relative-timestamp lines, template placeholders, zh editor tags, ru subscribe and
#       read-also chrome, ctrl+enter error-report widgets);
#   (c) a symbol and numeric junk rule: a whole line with no letters at all (photo
#       counters, vote counters, pagination digit runs, stray punctuation, a bare BOM).
# Precision still beats recall: exact or anchored whole-line matches only, the length gate
# stays, and content-bearing section headers seen in real review documents are excluded.

# Light normalization for the exact-inventory match: drop leading list-bullet and
# decoration markers and trailing colon, dot, arrow or dash decoration, then casefold.
# Keys are built with the same function at module load, so matching is symmetric.
_V2_LEAD_TRIM = re.compile(r"^[\s•·※＊*#▪◦●○►▶▲»›>+|=\-–—]+")
_V2_TAIL_TRIM = re.compile(r"[\s:：;；.。，,、…·•|<>›»←→+=\-–—]+$")


def _v2_key(line: str) -> str:
    return _V2_TAIL_TRIM.sub("", _V2_LEAD_TRIM.sub("", line)).casefold()


# The mined exact-line chrome inventory: lines that recur across many distinct documents
# of one language. Variants that normalize to the same key (case, trailing colon, leading
# bullet) are listed once.
_V2_EXACT_RAW: tuple[str, ...] = (
    # zh: nav, menu and feed chrome
    "我的頻道",
    "拖拉類別可自訂排序",
    "恢復預設 確定",
    "設定",
    "快訊",
    "出版：更新：",
    "發表評論",
    "相關新聞",
    "相关新闻",
    "東網電視",
    "更多新聞短片",
    "延伸閱讀",
    "分享",
    "分享給朋友",
    "分享让更多人看到",
    "上一則",
    "下一則",
    "更多",
    "超人氣",
    "回到最上面",
    "返回顶部",
    "上 / 下一篇新聞",
    "熱門新聞",
    "熱門快報",
    "最夯影音",
    "熱門行情",
    "人氣文章",
    "熱搜",
    "最近搜尋",
    "全部刪除",
    "點擊圖片放大",
    "字体大小",
    "小字号",
    "商品推薦",
    "udn討論區",
    "隱私權保護政策",
    "廣告",
    "广告",
    "我是廣告 請繼續往下閱讀",
    "[廣告]請繼續往下閱讀...",
    "關鍵字",
    "热词",
    "头条",
    "Tags",
    "#Hashtags",
    "See All",
    "SpotifyKKBOXSoundOnApple PodcastGoogle Podcast",
    "中央社",
    "人民网>>国际",
    "我們想讓你知道的是",
    # zh: social-follow, app-download and paywall prompts
    "緊貼財經時事新聞分析，讚好hket Facebook 專版",
    "LIKE我们的官方面簿网页以获取更多新信息",
    "etnet TV頻道現已登場，多元化財經及消閒影片輪流送上！ ► 即睇",
    "下载法广应用程序跟踪国际时事",
    "电邮新闻头条新闻就在您的每日新闻信里",
    "成为《华尔街日报》会员",
    "继续阅读全文，请订阅或登录",
    "畅读全文",
    "查看选项",
    "订户",
    # zh: consent and disclaimer walls
    "請細閱並示意接受以下私隱政策及免責聲明，按下「接受」表示你已同意並願意接受 am730 網站內之私隱政策及免責聲明。了解更多",
    "※【NOWnews 今日新聞】提醒您",
    "※以上言論不代表旺中媒體集團立場※",
    "＊《經濟通》所刊的署名及╱或不署名文章，相關內容屬作者個人意見，並不代表《經濟通》立",
    "場，《經濟通》所扮演的角色是提供一個自由言論平台。",
    # zh: a numbered site-nav block that appears verbatim in the zh documents
    "1. Inhalt",
    "2. Navigation",
    "3. Weitere Inhalte",
    "4. Metanavigation",
    "5. Suche",
    "6. Choose from 30 Languages",
    # ru: nav, feed and widget chrome
    "наверх",
    "вверх вверх",
    "картина дня",
    "вернуться к статье",
    "вернуться назад",
    "все новости",
    "показать ещё все новости",
    "загрузка",
    "новости",
    "новости партнеров",
    "новости сми2",
    "новости и материалы",
    "новости по теме",
    "лента новостей",
    "лента",
    "последние новости",
    "популярные новости",
    "популярное",
    "больше новостей",
    "топ-новости",
    "комментировать",
    "комментарии",
    "смотреть комментариикомментариев нет",
    "прослушать новость",
    "слушать новости",
    "озвучить текст",
    "остановить прослушивание",
    "прямой эфир",
    "сегодня в эфире",
    "сегодня в сми",
    "партнерский контент",
    "реклама",
    "поделиться",
    "подписаться",
    "подтвердить",
    "отправить",
    "закрыть",
    "просмотреть",
    "сначала новыесначала старые",
    "размер текста",
    "выделить главное",
    "вкл",
    "выкл",
    "перейти к основному содержанию",
    "данный сайт использует файлы cookies",
    "уведомления отключены",
    "больше не показывать",
    "прикрепить файл",
    "обратная связь",
    "форма обратной связи",
    "следите за новостями",
    "источник",
    "автор",
    "рейтинг",
    "опрос",
    "видео дня",
    "оценить новость",
    "лучшие букмекеры",
    "на евро-футболе",
    "читать ria.ru в",
    "карта сайта",
    "политика конфиденциальности - gdpr",
    "поддержать проект",
    "самое интересное - в нашем канале яндекс.дзен",
    "яндекс.дзен, telegram и viber!",
    "самое обсуждаемое",
    "заголовок открываемого материала",
    "на вашем ресурсе это будет выглядеть так",
    "чтобы разместить новость на сайте или в блоге скопируйте код",
    "регистрация пройдена успешно!",
    "пожалуйста, перейдите по ссылке из письма, отправленного на",
    "чтобы участвовать в дискуссии",
    "авторизуйтесь или зарегистрируйтесь",
    "выделите ошибку в тексте",
    "нашли ошибку?",
    "найдена ошибка?",
    "знайшли помилку в тексті?",
    "помилка",
    "смотреть карту",
    "карта городских событий",
    "пнвтсрчтптсбвс",
    "курсы валют",
    "к читателям",
    "рен тв новости",
    # ru: standalone section and rubric labels from news-site nav
    "общество",
    "происшествия",
    "политика",
    "экономика",
    "в мире",
    "спорт",
    "шоу-бизнес",
    "госорганы",
    # es: comment, share, subscribe and related-article chrome
    "comentarios",
    "mostrar comentarios",
    "comenta esta noticia",
    "comentar",
    "comenta",
    "escuchar",
    "escucha la radio en vivo",
    "escuchar este artículo",
    "reproducir",
    "publicidad",
    "compartir",
    "compartir el artículo",
    "compartir en",
    "compartir nota",
    "comparte",
    "comparte en",
    "suscríbete",
    "suscríbete a las notificaciones y entérate de todo",
    "suscríbete a las notificaciones y enterate de todo",
    "suscribite a noticias diarias",
    "contenido exclusivo para suscriptores digitales",
    "ver promos de suscripción",
    "¿ya eres suscriptor? inicia sesión aquí",
    "si ya eres suscriptor del impreso",
    "para continuar leyendo, suscríbete al acceso de contenidos web",
    "llegaste al límite de contenidos del mes",
    "crea una cuenta y podrás disfrutar de",
    "sabemos que te gusta estar siempre informado",
    "tenemos algo para ofrecerte",
    "síguenos en",
    "no te pierdas las últimas noticias",
    "te puede interesar",
    "también te puede interesar",
    "recomendado para vos",
    "video recomendado",
    "noticias relacionadas",
    "relacionadas",
    "temas relacionados",
    "artículo relacionados",
    "en esta nota",
    "lee también",
    "lea también",
    "además tenés que leer",
    "seguir leyendo",
    "para leer más",
    "saber más",
    "ver más",
    "temas",
    "etiquetas",
    "enlace copiado",
    "lo más leído",
    "lo más visto",
    "leídas",
    "últimas noticias",
    "último boletín",
    "en tendencia",
    "en directo",
    "en vivo",
    "cargando",
    "cargando más noticias",
    "cargando tendencia",
    "cargando contenido",
    "publicado el",
    "publicado",
    "actualizado",
    "redacción",
    "por",
    "foto",
    "fuente",
    "elige una ciudad",
    "selecciona tu región",
    "secciones",
    "descubre nuestras apps",
    "¿querés recibir notificaciones de alertas?",
    "¡bien! te has suscrito a notificaciones",
    "configura y elige tus preferencias",
    "activar",
    "actívate",
    "sigue bajando para encontrar más contenido",
    "llévatelo",
    "menéame",
    "códigos descuento",
    # es: standalone section labels
    "noticias",
    "deportes",
    "opinión",
    "nacional",
    "espectáculos",
    "edición impresa",
    # en: chrome the v1 rules miss
    "skip to main content",
    "breaking news",
    "news",
    "news alerts",
    "topics",
    "comments",
    "related to this story",
    "most popular",
    "recommended for you",
    "trending stories",
    "be the first to know",
    "log in",
    "edit",
    "filters",
    "you have permission to edit this article",
    "get up-to-the-minute news sent straight to your device",
    "get expert advice, insider tips and more",
    "want to cruise smarter?",
    "by proceeding, you agree to cruise critic’s privacy policy and terms of use",
)

_V2_EXACT: frozenset[str] = frozenset(k for k in map(_v2_key, _V2_EXACT_RAW) if k)


def _is_exact_chrome_v2(line: str) -> bool:
    return _v2_key(line) in _V2_EXACT


# Tight anchored patterns generalising a mined chrome family.
# "Comments / 0": a comment counter, never prose.
_V2_COMMENT_COUNT = _rx(r"^comments?\s*/\s*\d+$")
# "{{featured_button_text}}": an unexpanded template placeholder.
_V2_TEMPLATE_VAR = _rx(r"^\{\{[^{}]*\}\}$")
# Relative-timestamp lines: zh "1 小時前", en "9 days ago".
_V2_REL_TIME = _rx(
    r"^\d+\s*(?:小時前|小时前|分鐘前|分钟前|秒前|天前)$"
    r"|^\d+\s+(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?)\s+ago$"
)
# zh editor tags, e.g. "(责编：崔元苑、杨迪)".
_V2_ZH_EDITOR_TAG = _rx(r"^[（(](?:责编|責編)[:：][^（）()]{0,40}[)）]$")
# zh article-date widget, e.g. "文章日期：2021年08月03日".
_V2_ZH_ARTICLE_DATE = _rx(r"^文章日期[:：].{0,30}$")
# Traditional-script copyright footer marker; v1's _ARR covers only the simplified form.
_V2_ZH_COPYRIGHT = _rx(r"版權所有")
# ru subscribe calls to action: an anchored subscribe verb plus a social or newsletter
# token, so "Подписывайтесь на наш канал в Telegram" is stripped while prose such as
# "Подписаться на услугу можно через приложение банка." survives, having no token.
_V2_RU_SUBSCRIBE = _rx(
    r"^подпи(?:сывайтесь|саться|шитесь)\b"
    r"(?=.*(?:telegram|google|youtube|viber|instagram|\bvk\b|дзен|канал|новост|"
    r"рассылк|уведомлен|соцсет|\bнас\b|\blife\b)).{0,100}$"
)
# ru read-also and follow-us cross-promotion, e.g. "Читайте также:".
_V2_RU_READ_ALSO = _rx(r"^читайте\s+(?:также|нас)\b.{0,60}$")
# ru error-report widget: any short line wired to the Ctrl+Enter shortcut.
_V2_CTRL_ENTER = _rx(r"ctrl\s*\+\s*enter")


def _is_pattern_chrome_v2(line: str) -> bool:
    return bool(
        _V2_COMMENT_COUNT.match(line)
        or _V2_TEMPLATE_VAR.match(line)
        or _V2_REL_TIME.match(line)
        or _V2_ZH_EDITOR_TAG.match(line)
        or _V2_ZH_ARTICLE_DATE.match(line)
        or _V2_RU_SUBSCRIBE.match(line)
        or _V2_RU_READ_ALSO.match(line)
        or _V2_ZH_COPYRIGHT.search(line)
        or _V2_CTRL_ENTER.search(line)
    )


def _is_symbol_junk(line: str) -> bool:
    """Return True for a short line with no letters in any script, which is widget junk.

    This covers photo counters ("+2"), vote and comment counters ("0"), stray punctuation,
    calendar and pagination digit runs, separator runs and a bare byte-order mark. CJK
    ideographs count as letters under ``str.isalpha()``, so no prose line can fire.
    """
    return not any(ch.isalpha() for ch in line)


# The v2 rule set: all of v1, unchanged and still first, plus the multilingual rules.
BOILERPLATE_RULES_V2: tuple[_Rule, ...] = (
    *BOILERPLATE_RULES_V1,
    ("exact_chrome_v2", _is_exact_chrome_v2),
    ("pattern_chrome_v2", _is_pattern_chrome_v2),
    ("symbol_numeric_junk_v2", _is_symbol_junk),
)

_RULE_SETS: dict[str, tuple[_Rule, ...]] = {
    "v1": BOILERPLATE_RULES_V1,
    "v2": BOILERPLATE_RULES_V2,
}


def boilerplate_rules(version: str) -> tuple[_Rule, ...]:
    """Resolve a ``boilerplate_rules`` config version string to its concrete rule set."""
    try:
        return _RULE_SETS[version]
    except KeyError:
        raise ValueError(
            f"unknown boilerplate_rules version {version!r}; known: {sorted(_RULE_SETS)}"
        ) from None


def is_boilerplate(line: str, rules: tuple[_Rule, ...]) -> bool:
    """Return True if the already-stripped ``line`` is pure chrome under ``rules``.

    A line longer than :data:`_MAX_CHROME_LEN` is never chrome; otherwise it is boilerplate
    when some rule matches the whole line's chrome shape.
    """
    if not line or len(line) > _MAX_CHROME_LEN:
        return False
    return any(pred(line) for _, pred in rules)
