"""``BOILERPLATE_RULES_V2``: chrome stripping in zh, ru, es and en.

Every line in the strip list below was mined from the collection itself, 4000 documents per
language, and each carries the document frequency that put it there. That includes the zh
site navigation block that leaked verbatim into passages under v1.

The survive list is v1's whole prose battery plus native zh, ru and es prose that contains a
chrome word inside a real sentence. The rules favour precision, so prose is never stripped.
v1 stays byte-identical, because passages already built with it have to stay reproducible.
"""

from __future__ import annotations

import pytest

from ragtime.preprocess.boilerplate import (
    BOILERPLATE_RULES_V1,
    BOILERPLATE_RULES_V2,
    boilerplate_rules,
    is_boilerplate,
)
from ragtime.preprocess.chunk import chunk_document

pytestmark = pytest.mark.small


# --------------------------------------------------------------------------- #
# The registry, and v1 left as it was.
# --------------------------------------------------------------------------- #
def test_registry_resolves_v2_and_v1_is_untouched() -> None:
    assert boilerplate_rules("v2") is BOILERPLATE_RULES_V2
    assert boilerplate_rules("v1") is BOILERPLATE_RULES_V1
    # v1 still holds exactly its six rules, and v2 extends it, keeping the v1 rules first
    # and in order, rather than editing them.
    assert [name for name, _ in BOILERPLATE_RULES_V1] == [
        "nav_breadcrumb",
        "cookie_consent",
        "social_share",
        "dateline_byline",
        "copyright_footer",
        "read_more_pagination",
    ]
    assert BOILERPLATE_RULES_V2[: len(BOILERPLATE_RULES_V1)] == BOILERPLATE_RULES_V1


def test_v1_still_leaks_zh_nav_v2_catches_it() -> None:
    # v1 does not strip the zh navigation chrome, which is why it leaked, and v2 does.
    # This also stops v1 being "fixed" after the fact.
    assert not is_boilerplate("我的頻道", BOILERPLATE_RULES_V1)
    assert is_boilerplate("我的頻道", BOILERPLATE_RULES_V2)


# --------------------------------------------------------------------------- #
# Must strip: the mined chrome, each line with the document frequency that earned it.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "line",
    [
        # zh: the navigation block that leaked, line by line (df~301 each)
        "我的頻道",
        "* 拖拉類別可自訂排序",
        "恢復預設 確定",
        "設定",
        "快訊",
        # zh: feed/nav/share chrome
        "出版：更新：",  # df=392
        "發表評論...",  # df=386
        "相關新聞",  # df=386
        "相關新聞：",  # df=18 (trailing-colon variant, same key)
        "延伸閱讀",  # df=340
        "分享：",  # df=312
        "分享給朋友：",  # df=230
        "上一則",  # df=301
        "下一則",  # df=211
        "更多 >",  # df=299
        "回到最上面",  # df=230
        "上 / 下一篇新聞",  # df=227
        "熱門新聞",  # df=208
        "我是廣告 請繼續往下閱讀",  # df=200
        "字体大小:",  # df=150
        "LIKE我们的官方面簿网页以获取更多新信息",  # df=150
        "緊貼財經時事新聞分析，讚好hket Facebook 專版",  # df=248
        "廣告",  # df=33
        "广告",  # df=78
        "點擊圖片放大",  # df=66
        "※本文版權所有，非經授權，不得轉載。[ ETtoday著作權聲明 ]",  # df=209 (版權所有)
        "(责编：崔元苑、杨迪)",  # df=28 (editor-tag pattern)
        "(责编：任妍、李彤)",  # names not in the mined set: the pattern generalises
        "文章日期：2021年08月03日",  # df=13
        "1 小時前",  # df=41 (relative-time pattern)
        "继续阅读全文，请订阅或登录",  # df=24 (WSJ paywall)
        "人民网>>国际",  # df=35, a double-chevron breadcrumb v1's shape rule misses
        # ru: nav/widget/subscribe chrome
        "наверх",  # df=830
        "НАВЕРХ",  # df=29 (case-insensitive key)
        "Картина дня",  # df=830
        "Вернуться к статье",  # df=809
        "Все новости",  # df=325
        "Загрузка...",  # df=193
        "Лента новостей",  # df=127
        "Комментировать",  # df=112
        "Комментарии",  # df=105
        "Прослушать новость",  # df=72
        "Читайте также",  # df=64
        "ЧИТАЙТЕ ТАКЖЕ",  # df=37
        "Читайте также на Евро-Футболе:",  # df=28 (anchored pattern)
        "Читайте нас в Google Новости",  # df=44
        "Реклама",  # df=53
        "Поделиться:",  # df=49
        "Подписаться",  # df=48
        "Подписаться на Telegram-канал",  # df=51
        "Подписаться в Google News",  # df=51
        "Подписывайтесь на наш канал @gazeta.ru в Telegram",  # df=32
        "Подпишитесь на LIFE",  # df=39
        "Если вы нашли ошибку, пожалуйста, выделите фрагмент текста и нажмите Ctrl+Enter.",  # df=60
        "Ошибка в тексте? Выделите её и нажмите «Ctrl + Enter»",  # df=40
        "Нашли ошибку?",  # df=56
        "Размер текста:",  # df=60
        "Данный сайт использует файлы cookies",  # df=40 (ru cookie banner)
        "Партнерский контент",  # df=64
        "Новости партнеров",  # df=170
        "Больше не показывать",  # df=37
        "Следите за новостями:",  # df=37
        "Источник:",  # df=34
        "ПнВтСрЧтПтСбВс",  # df=29 (calendar row)
        # es: comment/share/subscribe/related chrome
        "Comentarios",  # df=187
        "• Escuchar",  # df=174 (bullet-prefixed key)
        "Publicidad",  # df=165
        "PUBLICIDAD",  # df=31
        "- Publicidad -",  # df=31
        "Compartir",  # df=121
        "Compartir el artículo",  # df=46
        "Suscríbete",  # df=82
        "Suscríbete a las notificaciones y entérate de todo",  # df=33
        "SUSCRIBITE A NOTICIAS DIARIAS",  # df=20
        "Contenido exclusivo para suscriptores digitales",  # df=46
        "TE PUEDE INTERESAR",  # df=78
        "Te puede interesar:",  # df=15
        "También te puede interesar",  # df=22
        "Noticias relacionadas",  # df=48
        "Síguenos en:",  # df=41
        "Enlace copiado",  # df=42
        "Lo más leído",  # df=28
        "SEGUIR LEYENDO:",  # df=26
        "Lee También",  # df=28
        "Cargando más noticias...",  # df=31
        "Etiquetas",  # df=67
        "Temas:",  # df=23
        "• REDACCIÓN",  # df=89
        "Mostrar comentarios",  # df=49
        # en: chrome that v1 misses
        "Skip to main content",  # df=145
        "Comments / 0",  # df=232 (comment-counter pattern)
        "Comments / 12",  # pattern generalizes to any count
        "Related to this story",  # df=104
        "Most Popular",  # df=97
        "Recommended for you",  # df=88
        "Trending Stories",  # df=48
        "Be the first to know",  # df=66
        "You have permission to edit this article.",  # df=108
        "{{featured_button_text}}",  # df=48 (template-placeholder pattern)
        "9 days ago",  # df=56 (relative-time pattern)
        "Breaking News",  # df=129
        # symbol/numeric junk (no letters in any script)
        "+2",  # zh photo counter, df=70
        "0",  # a vote or comment counter: ru df=187, en df=136, es df=80
        ",,",  # ru df=500
        "16+",  # ru df=44
        "9101112131415",  # ru calendar digit run, df=30
        "＝＝＝＝＝",  # zh separator run, df=13
        "\ufeff",  # ru BOM line, df=81
    ],
)
def test_v2_strips_mined_chrome(line: str) -> None:
    assert is_boilerplate(line, BOILERPLATE_RULES_V2)


# --------------------------------------------------------------------------- #
# Must survive: v1's whole prose battery, plus native prose carrying a chrome word inside
# a real sentence. v2 strips none of these.
# --------------------------------------------------------------------------- #
_V1_PROSE = [
    "Home sales rose 8%/10% year-over-year in the two largest metro markets.",
    "Start-up costs rose from 20/30 percent to 45/60 percent among small businesses.",
    (
        "A federal judge ruled the AI company violated copyright law by training on "
        "the artist's work without a license."
    ),
    "The bill has three main clauses: (a) taxation, (b) benefits, and (c) enforcement.",
    (
        "The EU cookie consent rules take effect Monday, forcing sites to ask "
        "permission before tracking users."
    ),
    "The government updated its cookie policy after a public consultation this year.",
    "By Monday the deal between the two firms had already been signed.",
    "The council will share the plan with residents before the vote next week.",
    "In 2024 the company reported record revenue across every regional market.",
    "The minister announced a new policy today.",
]

_NATIVE_PROSE = [
    # zh: 分享, 設定, 廣告 and 更多 inside real sentences
    "该公司周五在社交媒体上分享了季度业绩报告。",
    "他表示，新設定的減碳目標將在明年之前實現。",
    "廣告收入在第三季度增長了百分之十，超出分析師預期。",
    "政府計劃投入更多資源改善公共醫療系統。",
    "委員會下週將發表評論報告，說明調查的初步結果。",
    # ru: подписать, реклама, новости and общество inside real sentences
    "Компания подписала соглашение о сотрудничестве в пятницу.",
    "Подписаться на услугу можно через мобильное приложение банка.",
    "Реклама на телевидении подорожала на десять процентов в этом году.",
    "Общество постепенно привыкает к новым санитарным правилам.",
    "Все новости о переговорах публиковались с задержкой в несколько часов.",
    # es: compartir, suscribir, publicidad and redacción inside real sentences
    "La empresa compartió los resultados trimestrales con los inversores el viernes.",
    "Suscribieron el convenio los ministros de ambos países tras la cumbre.",
    "La publicidad digital superó por primera vez a la televisión en ingresos.",
    "Los periodistas de la redacción publicaron una investigación sobre el caso.",
    "El gobierno nacional anunció nuevas medidas económicas para el sector.",
    # en: chrome words opening a sentence, which must not match the exact keys
    "Breaking news coverage of the storm continued through the night on every channel.",
    "Comments from the mayor drew criticism from opposition councillors.",
    "News of the merger sent both companies' shares sharply higher.",
]


@pytest.mark.parametrize("line", _V1_PROSE + _NATIVE_PROSE)
def test_v2_keeps_real_prose(line: str) -> None:
    assert not is_boilerplate(line, BOILERPLATE_RULES_V2)


def test_long_line_gate_still_holds_under_v2() -> None:
    # The 120-character guard still applies under v2: a long line that opens with chrome
    # words is prose.
    long_prose = "Реклама и маркетинг " + ("и аналитика " * 12) + "остаются главными темами."
    assert len(long_prose) > 120
    assert not is_boilerplate(long_prose, BOILERPLATE_RULES_V2)


# --------------------------------------------------------------------------- #
# End to end: under v2 the zh navigation block is gone from the chunked passages and the
# real zh content survives. This is the leak itself, closed.
# --------------------------------------------------------------------------- #
def _zh_doc() -> dict:
    return {
        "id": "zho-docs/0000200",
        "text": (
            "我的頻道\n"
            "* 拖拉類別可自訂排序\n"
            "恢復預設 確定\n"
            "設定\n"
            "快訊\n"
            "\n"
            "巴西女子網球選手在東京奧運女子雙打勇奪銅牌。\n"
            "\n"
            "※本文版權所有，非經授權，不得轉載。"
        ),
        "url": "u",
        "date": "d",
        "lang": "zh",
    }


def test_zh_nav_block_absent_from_v2_passages(fake_segmenter, fake_tokenizer) -> None:
    passages = chunk_document(
        _zh_doc(), fake_segmenter, fake_tokenizer,
        token_budget=512, overlap_frac=0.0, boilerplate_rules_version="v2",
    )
    joined = " ".join(p.text for p in passages)
    assert "勇奪銅牌" in joined  # real content survives
    for chrome in ("我的頻道", "拖拉類別", "恢復預設", "設定", "快訊", "版權所有"):
        assert chrome not in joined


def test_zh_nav_block_leaks_under_v1(fake_segmenter, fake_tokenizer) -> None:
    # Under v1 the same document leaks its navigation chrome, which is why v2 exists.
    # v1's behaviour is pinned here so it cannot change quietly.
    passages = chunk_document(
        _zh_doc(), fake_segmenter, fake_tokenizer,
        token_budget=512, overlap_frac=0.0, boilerplate_rules_version="v1",
    )
    joined = " ".join(p.text for p in passages)
    assert "我的頻道" in joined
