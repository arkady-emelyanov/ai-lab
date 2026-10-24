/*
 * A small JSON reader and writer for the container tools (CDI
 * specifications, OCI bundles' config.json): a tree of values, objects
 * keeping their keys in order and numbers as their original text, so a
 * document read and written again changes only where it was edited.
 */
#ifndef FAKEGPU_JSON_H
#define FAKEGPU_JSON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum json_type { JSON_NULL, JSON_BOOL, JSON_NUMBER, JSON_STRING, JSON_ARRAY, JSON_OBJECT };

typedef struct json {
    enum json_type type;
    int boolean;
    char *text;            /* string (decoded) or number (as written) */
    struct json **items;   /* array elements or object values */
    char **keys;           /* object keys */
    size_t n, cap;
} json_t;

static json_t *json_new(enum json_type type)
{
    json_t *v = calloc(1, sizeof *v);
    if (!v)
        abort();
    v->type = type;
    return v;
}

static json_t *json_string(const char *s)
{
    json_t *v = json_new(JSON_STRING);
    v->text = strdup(s);
    return v;
}

static void json_push(json_t *v, const char *key, json_t *item)
{
    if (v->n == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 8;
        v->items = realloc(v->items, v->cap * sizeof *v->items);
        if (v->type == JSON_OBJECT)
            v->keys = realloc(v->keys, v->cap * sizeof *v->keys);
        if (!v->items || (v->type == JSON_OBJECT && !v->keys))
            abort();
    }
    if (v->type == JSON_OBJECT)
        v->keys[v->n] = strdup(key);
    v->items[v->n++] = item;
}

/* An object's value for key, or NULL. */
static json_t *json_get(const json_t *v, const char *key)
{
    if (!v || v->type != JSON_OBJECT)
        return NULL;
    for (size_t i = 0; i < v->n; i++)
        if (!strcmp(v->keys[i], key))
            return v->items[i];
    return NULL;
}

/* An object's value for key, made with type if it is missing. */
static json_t *json_child(json_t *v, const char *key, enum json_type type)
{
    json_t *c = json_get(v, key);
    if (!c) {
        c = json_new(type);
        json_push(v, key, c);
    }
    return c;
}

static const char *json_str(const json_t *v) { return v && v->type == JSON_STRING ? v->text : NULL; }

/* ---- reading ---------------------------------------------------------------------- */

typedef struct { const char *p; int err; } json_in;

static void json_ws(json_in *in)
{
    while (*in->p == ' ' || *in->p == '\t' || *in->p == '\n' || *in->p == '\r')
        in->p++;
}

static void json_utf8(char **out, unsigned cp)
{
    char *o = *out;
    if (cp < 0x80)
        *o++ = (char)cp;
    else if (cp < 0x800)
        *o++ = (char)(0xC0 | cp >> 6), *o++ = (char)(0x80 | (cp & 0x3F));
    else if (cp < 0x10000)
        *o++ = (char)(0xE0 | cp >> 12), *o++ = (char)(0x80 | (cp >> 6 & 0x3F)), *o++ = (char)(0x80 | (cp & 0x3F));
    else
        *o++ = (char)(0xF0 | cp >> 18), *o++ = (char)(0x80 | (cp >> 12 & 0x3F)),
        *o++ = (char)(0x80 | (cp >> 6 & 0x3F)), *o++ = (char)(0x80 | (cp & 0x3F));
    *out = o;
}

static unsigned json_hex4(json_in *in)
{
    unsigned v = 0;
    for (int i = 0; i < 4; i++) {
        char c = *in->p++;
        v <<= 4;
        if (c >= '0' && c <= '9')
            v |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f')
            v |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F')
            v |= (unsigned)(c - 'A' + 10);
        else
            in->err = 1;
    }
    return v;
}

static char *json_read_string(json_in *in)
{
    if (*in->p != '"') {
        in->err = 1;
        return NULL;
    }
    in->p++;
    char *buf = malloc(strlen(in->p) + 1), *o = buf;
    if (!buf)
        abort();
    while (*in->p && *in->p != '"') {
        if (*in->p != '\\') {
            *o++ = *in->p++;
            continue;
        }
        in->p++;
        char c = *in->p++;
        switch (c) {
        case 'n': *o++ = '\n'; break;
        case 't': *o++ = '\t'; break;
        case 'r': *o++ = '\r'; break;
        case 'b': *o++ = '\b'; break;
        case 'f': *o++ = '\f'; break;
        case 'u': {
            unsigned cp = json_hex4(in);
            if (cp >= 0xD800 && cp < 0xDC00 && in->p[0] == '\\' && in->p[1] == 'u') {
                in->p += 2;
                unsigned lo = json_hex4(in);
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            }
            json_utf8(&o, cp);
            break;
        }
        case 0: in->err = 1; in->p--; break;
        default: *o++ = c; break; /* \" \\ \/ */
        }
    }
    if (*in->p != '"')
        in->err = 1;
    else
        in->p++;
    *o = 0;
    return buf;
}

static json_t *json_read_value(json_in *in, int depth)
{
    json_ws(in);
    if (depth > 128) {
        in->err = 1;
        return NULL;
    }
    char c = *in->p;
    if (c == '{' || c == '[') {
        json_t *v = json_new(c == '{' ? JSON_OBJECT : JSON_ARRAY);
        char end = c == '{' ? '}' : ']';
        in->p++;
        json_ws(in);
        if (*in->p == end) {
            in->p++;
            return v;
        }
        while (!in->err) {
            char *key = NULL;
            if (v->type == JSON_OBJECT) {
                json_ws(in);
                key = json_read_string(in);
                json_ws(in);
                if (*in->p != ':')
                    in->err = 1;
                else
                    in->p++;
            }
            json_t *item = in->err ? NULL : json_read_value(in, depth + 1);
            if (in->err) {
                free(key);
                break;
            }
            json_push(v, key, item);
            free(key);
            json_ws(in);
            if (*in->p == ',') {
                in->p++;
                continue;
            }
            if (*in->p == end) {
                in->p++;
                break;
            }
            in->err = 1;
        }
        return v;
    }
    if (c == '"') {
        json_t *v = json_new(JSON_STRING);
        v->text = json_read_string(in);
        return v;
    }
    if (!strncmp(in->p, "true", 4) || !strncmp(in->p, "false", 5)) {
        json_t *v = json_new(JSON_BOOL);
        v->boolean = c == 't';
        in->p += v->boolean ? 4 : 5;
        return v;
    }
    if (!strncmp(in->p, "null", 4)) {
        in->p += 4;
        return json_new(JSON_NULL);
    }
    const char *start = in->p;
    while (*in->p && strchr("+-0123456789.eE", *in->p))
        in->p++;
    if (in->p == start) {
        in->err = 1;
        return NULL;
    }
    json_t *v = json_new(JSON_NUMBER);
    v->text = strndup(start, (size_t)(in->p - start));
    return v;
}

/* The document in text, or NULL if it is not JSON. */
static json_t *json_parse(const char *text)
{
    json_in in = {text, 0};
    json_t *v = json_read_value(&in, 0);
    json_ws(&in);
    return in.err || *in.p ? NULL : v;
}

/* The JSON document in a file, or NULL (errno set when it cannot be read). */
static json_t *json_load(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f)
        return NULL;
    size_t len = 0, cap = 65536;
    char *buf = malloc(cap);
    size_t got;
    while (buf && (got = fread(buf + len, 1, cap - len - 1, f)) > 0) {
        len += got;
        if (cap - len < 2) {
            cap *= 2;
            buf = realloc(buf, cap);
        }
    }
    fclose(f);
    if (!buf)
        abort();
    buf[len] = 0;
    json_t *v = json_parse(buf);
    free(buf);
    return v;
}

/* ---- writing ------------------------------------------------------------------------- */

static void json_write_string(FILE *f, const char *s)
{
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p == '"' || *p == '\\')
            fprintf(f, "\\%c", *p);
        else if (*p == '\n')
            fputs("\\n", f);
        else if (*p == '\t')
            fputs("\\t", f);
        else if (*p == '\r')
            fputs("\\r", f);
        else if (*p < 0x20)
            fprintf(f, "\\u%04x", *p);
        else
            fputc(*p, f);
    }
    fputc('"', f);
}

static void json_write(FILE *f, const json_t *v)
{
    switch (v->type) {
    case JSON_NULL: fputs("null", f); break;
    case JSON_BOOL: fputs(v->boolean ? "true" : "false", f); break;
    case JSON_NUMBER: fputs(v->text, f); break;
    case JSON_STRING: json_write_string(f, v->text); break;
    case JSON_ARRAY:
    case JSON_OBJECT:
        fputc(v->type == JSON_ARRAY ? '[' : '{', f);
        for (size_t i = 0; i < v->n; i++) {
            if (i)
                fputc(',', f);
            if (v->type == JSON_OBJECT) {
                json_write_string(f, v->keys[i]);
                fputc(':', f);
            }
            json_write(f, v->items[i]);
        }
        fputc(v->type == JSON_ARRAY ? ']' : '}', f);
        break;
    }
}

#endif
