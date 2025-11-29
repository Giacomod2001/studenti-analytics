# ─── 7.4) SELEZIONE DELLA TABELLA ─────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Seleziona Tabella")
    table_names = [t["name"] for t in tables_info]
    sel_table = st.sidebar.selectbox("Scegli la tabella da analizzare:", options=table_names)

    current_info = next((t for t in tables_info if t["name"] == sel_table), None)

    # ─── 7.5) CARICAMENTO DEI DATI (TUTTI) ────────────────────────────────────────
    with st.spinner(f"⏳ Caricamento dati da {sel_table}..."):
        df_dict, load_msg = load_table_data(sel_table)

    if df_dict is None:
        st.sidebar.error(load_msg)
        st.error(f"❌ Errore: {load_msg}")
        st.stop()
    else:
        st.sidebar.success(load_msg)
        df = dict_to_dataframe(df_dict)

    # ─── 7.6) PULSANTE DI REFRESH DELLA CACHE ─────────────────────────────────────
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Ricarica Dati (cancella cache)"):
        st.cache_data.clear()
        st.rerun()

    # ─── 7.7) RENDERING DELL'ANALISI PRINCIPALE ──────────────────────────────────
    render_table_inspection(df, current_info)


if name == "main":
    main()
