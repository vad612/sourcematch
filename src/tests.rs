use crate::{
    economics::{self, Market, Range},
    input::{self, ImportStats, Listing, Offer},
    models::{EmbeddingProfile, ModelClient, RerankingProfile},
    normalize::{self, Lexicon},
    pipeline::{CatalogIndex, Context},
    search::{Hit, Index},
};
use rust_decimal::Decimal;
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::TcpListener,
    path::Path,
    thread,
};

fn lexicon() -> Lexicon {
    Lexicon::read(include_str!("../rules/products.tsv")).unwrap()
}

fn market() -> Market {
    Market::load(Path::new("rules/uzbekistan.toml")).unwrap()
}

fn d(n: &str) -> Decimal {
    n.parse().unwrap()
}

fn listing() -> Listing {
    Listing {
        id: "listing".into(),
        title: "Ваза, стекло, 20 см, круглый, серия Классик".into(),
        facts: lexicon().facts("Ваза, стекло, 20 см, круглый, серия Классик", "", None),
        price: Some(d("125000")),
        demand: [Some(d("70")), None],
    }
}

fn offer(id: &str, quote: &str) -> Offer {
    Offer {
        id: id.into(),
        title: "玻璃经典款圆形20厘米花瓶".into(),
        facts: lexicon().facts("玻璃经典款圆形20厘米花瓶", "", Some("件")),
        moq: Some(1),
        price_tiers: quote.into(),
        weight: "0.25kg".into(),
        package: "20x10x5cm".into(),
    }
}

#[test]
fn units_are_canonical_and_colour_is_free() {
    let facts = lexicon().facts(
        "Fotoramka, yog’och, 10x15 sm, devorga osiladigan, Klassik, ko’k, 3 dona",
        "",
        None,
    );
    let alternate = lexicon().facts("红色、经典款壁挂式相框15*10厘米实木", "", Some("件"));
    assert_eq!(facts.group(), alternate.group());
    assert_eq!(facts.measures, alternate.measures);
    assert_eq!(facts.traits, alternate.traits);
    assert_eq!(facts.units, Some(3));
    assert_eq!(
        normalize::packed_weight("外箱20个/箱 毛重12.0kg"),
        Some((d("12"), Some(20)))
    );
    assert_eq!(normalize::packed_weight("0.4斤"), Some((d("0.2"), None)));
    assert_eq!(normalize::packed_weight("毛重12kg"), None);
}

#[test]
fn price_tiers_and_pack_quantities_use_supplier_units() {
    let market = market();
    let mut offer = offer("cheap", "1-1199件:¥5|≥1200件:¥2");
    offer.moq = Some(1_200);
    let mut listing = listing();
    listing.facts.units = Some(3);
    let purchase = economics::purchase(0, &listing.facts, &offer, &market);
    assert_eq!(purchase.eligible, Some(true));
    assert_eq!(purchase.supplier_units, Some(1_200));
    assert_eq!(purchase.price, Some(Range::exact(d("6"))));
    offer.moq = Some(3_001);
    assert_eq!(
        economics::purchase(0, &listing.facts, &offer, &market).eligible,
        Some(false)
    );
    assert_eq!(
        economics::price_at("¥0.82-0.90", 300),
        Some(Range {
            low: d("0.82"),
            high: d("0.90")
        })
    );
    assert_eq!(economics::price_at("1-20件: ¥1 | 20-100件: ¥2", 20), None);
    assert_eq!(
        economics::price_at("10个起批 ¥2.94; 1000个以上 ¥2.21", 300),
        Some(Range::exact(d("2.94")))
    );
    assert_eq!(
        economics::price_at("50-49套:¥22.04|50-1999套:¥21.82|≥2000套:¥20.78", 300),
        Some(Range::exact(d("21.82")))
    );
}

#[test]
fn costs_match_independent_reference_and_unknown_weight_is_blank() {
    let market = market();
    let listing = listing();
    let mut offer = offer("supplier", "¥2");
    let shipping = economics::logistics(&listing.facts, &offer, None);
    let quote = Some(Range::exact(d("2")));
    let costs = economics::economics(&listing, quote, &shipping, &market).unwrap();
    assert_eq!(costs.landed, Some(Range::exact(d("17955.84"))));
    assert_eq!(costs.margin, Some(Range::exact(d("70794.16"))));
    offer.weight.clear();
    let missing = economics::logistics(&listing.facts, &offer, None);
    let bounds = economics::economics(&listing, quote, &missing, &market).unwrap();
    assert!(bounds.landed.is_none() && bounds.margin.is_none() && bounds.margin_ceiling.is_some());
    for (liters, charge) in [
        ("0.599", "6500"),
        ("0.6", "7500"),
        ("5", "15000"),
        ("37", "15000"),
        ("37.001", "32000"),
    ] {
        assert_eq!(market.fulfillment_charge(d(liters)), Some(d(charge)));
    }
}

#[test]
fn cheapest_confirmed_offer_is_first_even_outside_hybrid_top_three() {
    let market = market();
    let offers = [
        offer("A", "¥9"),
        offer("B", "¥8"),
        offer("C", "¥7"),
        offer("D", "¥2"),
    ];
    let catalog = CatalogIndex::new(&[]);
    let compute = Context {
        offers: &offers,
        market: &market,
        catalog: &catalog,
        reranker: None,
    };
    let hits = (0..offers.len())
        .map(|offer| Hit {
            offer,
            group: true,
            dense_rank: Some(offer + 1),
            ..Hit::default()
        })
        .collect();
    let row = compute
        .process("listing", Some(&listing()), hits, 0.)
        .unwrap()
        .output;
    assert_eq!(row.offer_ids, "D;A;B");
    assert_eq!(row.decision, "брать");
    assert_eq!(row.unit_margin_uzs, "70794.16");
}

#[test]
fn certification_is_independent_from_commission_and_unknown_demand() {
    let mut market = market();
    market.certified.push("Декор".into());
    market.commissions.remove("Декор");
    let mut listing = listing();
    listing.demand = [None; 2];
    let offers = [offer("supplier", "¥2")];
    let catalog = CatalogIndex::new(&[]);
    let compute = Context {
        offers: &offers,
        market: &market,
        catalog: &catalog,
        reranker: None,
    };
    let row = compute
        .process("listing", Some(&listing), vec![Hit::default()], 0.)
        .unwrap()
        .output;
    assert_eq!(row.decision, "не брать");
    assert!(row.reason.starts_with("CERT"));
    assert_eq!(row.landed_cost_uzs, "17955.84");
    assert!(row.unit_margin_uzs.is_empty());
}

#[test]
fn conflicting_rows_do_not_affect_other_cards() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("listing.csv");
    std::fs::write(&path, "title,listing_id,price_uzs,est_orders_b,extra\nВаза,A,50000,50,1\nВаза,A,60000,,2\nПлед,B,70000,75,3\n").unwrap();
    let mut stats = ImportStats::default();
    let rows = input::listings(&path, &lexicon(), &mut stats, &Default::default()).unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].price, None);
    assert_eq!(rows[0].demand, [None, Some(d("50"))]);
    assert_eq!(rows[1].price, Some(d("70000")));
    assert_eq!(stats.duplicate_ids, 1);
}

#[test]
fn hybrid_results_keep_exact_group_and_rank_dense() {
    let offers = [offer("A", "¥9"), offer("B", "¥8")];
    let index = Index::new(&offers, vec![vec![1., 0.], vec![0., 1.]], 2).unwrap();
    let scores = index.dense_scores(vec![vec![0., 1.]]).unwrap();
    let hits = index
        .get(
            &listing().title,
            &listing().facts,
            scores.row(0).as_slice().unwrap(),
            true,
            1,
        )
        .unwrap();
    assert_eq!(
        hits.iter().find(|h| h.dense_rank == Some(1)).unwrap().offer,
        1
    );
    assert!(hits.iter().any(|h| h.offer == 0 && h.group));
    assert!(hits.iter().any(|h| h.lexical_rank.is_some()));
}

fn one_response(reply: &'static str) -> (String, thread::JoinHandle<serde_json::Value>) {
    let server = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let endpoint = format!("http://{}/test", server.local_addr().unwrap());
    let thread = thread::spawn(move || {
        let (mut socket, _) = server.accept().unwrap();
        socket
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        let mut reader = BufReader::new(&mut socket);
        let mut length = 0;
        loop {
            let mut line = String::new();
            assert_ne!(reader.read_line(&mut line).unwrap(), 0);
            if line == "\r\n" {
                break;
            }
            if let Some((name, value)) = line.split_once(':')
                && name.eq_ignore_ascii_case("content-length")
            {
                length = value.trim().parse().unwrap();
            }
        }
        let mut body = vec![0; length];
        reader.read_exact(&mut body).unwrap();
        write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", reply.len(), reply).unwrap();
        serde_json::from_slice(&body).unwrap()
    });
    (endpoint, thread)
}

#[test]
fn local_model_contract_restores_indices_and_rejects_invalid_replies() {
    let (url, request) =
        one_response(r#"{"data":[{"index":1,"embedding":[0,1]},{"index":0,"embedding":[1,0]}]}"#);
    let model = ModelClient::new().unwrap();
    let profile = EmbeddingProfile {
        url,
        model: "fixture-embedding".into(),
        dimension: 2,
        revision: "fixture".into(),
        batch_size: 2,
        check_loaded: false,
    };
    assert_eq!(
        model.embed(&profile, &["one", "two"]).unwrap(),
        vec![vec![1., 0.], vec![0., 1.]]
    );
    assert_eq!(request.join().unwrap()["model"], "fixture-embedding");
    let (url, request) = one_response(
        r#"{"results":[{"index":1,"relevance_score":0.8},{"index":0,"relevance_score":0.2}]}"#,
    );
    let mut profile = RerankingProfile {
        url,
        model: "fixture-reranker".into(),
        shortlist: 2,
    };
    assert_eq!(
        model.rerank(&profile, "query", &["one", "two"]).unwrap(),
        vec![0.2, 0.8]
    );
    assert_eq!(request.join().unwrap()["top_n"], 2);
    let (url, request) = one_response(r#"{"error":"Unexpected endpoint or method"}"#);
    profile.url = url;
    assert!(
        model
            .rerank(&profile, "query", &["one"])
            .unwrap_err()
            .to_string()
            .contains("Unexpected endpoint")
    );
    request.join().unwrap();
}

proptest::proptest! {
    #[test]
    fn rising_exchange_rate_never_improves_margin(rate in 1000u32..50000, delta in 1u32..5000) {
        let mut market = market();
        market.exchange_rate = Decimal::from(rate);
        let listing = listing();
        let shipping = economics::logistics(&listing.facts, &offer("A", "¥2"), None);
        let quote = Some(Range::exact(d("2")));
        let before = economics::economics(&listing, quote, &shipping, &market).unwrap().margin.unwrap();
        market.exchange_rate += Decimal::from(delta);
        let after = economics::economics(&listing, quote, &shipping, &market).unwrap().margin.unwrap();
        proptest::prop_assert!(after.high < before.low);
    }
}

#[test]
fn full_cli_uses_reranking_and_keeps_economics_offer_first() {
    let server = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let base = format!("http://{}", server.local_addr().unwrap());
    let test_dir = tempfile::tempdir().unwrap();
    let root = test_dir.path();

    let replies = thread::spawn(move || {
        let mut rerank_requests = 0;
        for _ in 0..5 {
            let (mut socket, _) = server.accept().unwrap();
            socket
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            let mut reader = BufReader::new(&mut socket);
            let mut first_line = String::new();
            reader.read_line(&mut first_line).unwrap();
            let length = loop {
                let mut line = String::new();
                assert_ne!(reader.read_line(&mut line).unwrap(), 0);
                if let Some((name, value)) = line.split_once(':')
                    && name.eq_ignore_ascii_case("content-length")
                {
                    break value.trim().parse::<usize>().unwrap();
                }
            };
            loop {
                let mut line = String::new();
                assert_ne!(reader.read_line(&mut line).unwrap(), 0);
                if line == "\r\n" {
                    break;
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            let request: serde_json::Value = serde_json::from_slice(&body).unwrap();
            let reply = if first_line.contains("/embed") {
                let count = request["input"].as_array().unwrap().len();
                serde_json::json!({"data": (0..count).map(|index| serde_json::json!({"index":index,"embedding":[1.,0.]})).collect::<Vec<_>>()})
            } else {
                rerank_requests += 1;
                let count = request["documents"].as_array().unwrap().len();
                serde_json::json!({"results": (0..count).rev().map(|index| serde_json::json!({"index":index,"relevance_score": if index == 0 {0.9} else {0.1}})).collect::<Vec<_>>()})
            }.to_string();
            write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", reply.len(), reply).unwrap();
        }
        rerank_requests
    });

    let listings = root.join("listings.csv");
    let offers = root.join("offers.csv");
    let profile = root.join("profile.toml");
    let input_profile = root.join("input-profile.toml");
    let output = root.join("submission.csv");
    std::fs::write(&listings, "market_id,title,price_uzs,est_orders_b\nL,\"Ваза, стекло, 20 см, круглый, серия Классик\",125000,70\n").unwrap();
    std::fs::write(&input_profile, "[listings]\nmarket_id = \"listing_id\"\n").unwrap();
    std::fs::write(&offers, "offer_id,title_cn,unit,moq,price_tiers,weight,package_size\nA,玻璃经典款圆形20厘米花瓶,件,1,¥4,0.25kg,20x10x5cm\nB,玻璃经典款圆形20厘米花瓶,件,1,¥2,0.25kg,20x10x5cm\n").unwrap();
    std::fs::write(&profile, format!("[embedding]\nurl=\"{base}/embed\"\nmodel=\"embedding\"\ndimension=2\nrevision=\"fixture\"\nbatch_size=32\n[reranking]\nurl=\"{base}/rerank\"\nmodel=\"reranker\"\nshortlist=12\n[search]\ncandidates=16\nexpanded_candidates=32\n")).unwrap();

    crate::run(crate::Run {
        listings,
        offers,
        catalog: None,
        template: None,
        input_profile: Some(input_profile),
        rules: Path::new("rules/uzbekistan.toml").into(),
        lexicon: Path::new("rules/products.tsv").into(),
        search_profile: profile,
        out: output.clone(),
        runs_dir: root.join("runs"),
        workers: 2,
        no_rerank: false,
    })
    .unwrap();
    let rows: Vec<crate::pipeline::Output> = csv::Reader::from_path(output)
        .unwrap()
        .deserialize()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].offer_ids, "B;A");
    assert_eq!(rows[0].decision, "брать");
    assert_eq!(rows[0].unit_margin_uzs, "70794.16");
    assert_eq!(replies.join().unwrap(), 2);
}
