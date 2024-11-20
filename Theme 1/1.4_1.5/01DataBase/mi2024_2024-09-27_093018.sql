--
-- PostgreSQL database dump
--

-- Dumped from database version 16.0
-- Dumped by pg_dump version 16.0

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: document_types; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.document_types (
    type_id integer NOT NULL,
    type_name character varying(100) NOT NULL
);


ALTER TABLE public.document_types OWNER TO postgres;

--
-- Name: document_types_type_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.document_types_type_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.document_types_type_id_seq OWNER TO postgres;

--
-- Name: document_types_type_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.document_types_type_id_seq OWNED BY public.document_types.type_id;


--
-- Name: documents; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.documents (
    document_id integer NOT NULL,
    number character varying(50) NOT NULL,
    type_id integer,
    content text NOT NULL,
    date date NOT NULL,
    completion_status character varying(20),
    CONSTRAINT documents_completion_status_check CHECK (((completion_status)::text = ANY ((ARRAY['виконано'::character varying, 'в процесі'::character varying, 'не виконано'::character varying])::text[])))
);


ALTER TABLE public.documents OWNER TO postgres;

--
-- Name: documents_document_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.documents_document_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.documents_document_id_seq OWNER TO postgres;

--
-- Name: documents_document_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.documents_document_id_seq OWNED BY public.documents.document_id;


--
-- Name: document_types type_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.document_types ALTER COLUMN type_id SET DEFAULT nextval('public.document_types_type_id_seq'::regclass);


--
-- Name: documents document_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.documents ALTER COLUMN document_id SET DEFAULT nextval('public.documents_document_id_seq'::regclass);


--
-- Data for Name: document_types; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.document_types (type_id, type_name) FROM stdin;
1	Наказ
2	Звіт
3	Лист
\.


--
-- Data for Name: documents; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.documents (document_id, number, type_id, content, date, completion_status) FROM stdin;
1	123/45	1	Наказ про організацію заходів...	2024-09-27	в процесі
\.


--
-- Name: document_types_type_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.document_types_type_id_seq', 3, true);


--
-- Name: documents_document_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.documents_document_id_seq', 1, true);


--
-- Name: document_types document_types_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.document_types
    ADD CONSTRAINT document_types_pkey PRIMARY KEY (type_id);


--
-- Name: documents documents_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.documents
    ADD CONSTRAINT documents_pkey PRIMARY KEY (document_id);


--
-- Name: documents documents_type_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.documents
    ADD CONSTRAINT documents_type_id_fkey FOREIGN KEY (type_id) REFERENCES public.document_types(type_id);


--
-- PostgreSQL database dump complete
--

